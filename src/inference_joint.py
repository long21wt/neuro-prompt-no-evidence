import argparse
import base64
import io
import json
import os
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    set_seed,
)
from qwen_vl_utils import process_vision_info


SYSTEM_PROMPT = "You are a helpful medical assistant in clinical psychiatry."


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_mri_content(txt_filename, mri_base_path):
    pid = os.path.splitext(txt_filename)[0]
    sub_dir = os.path.join(mri_base_path, f"sub-{int(pid):04d}")
    if not os.path.exists(sub_dir):
        return [{"type": "text", "text": f"No MRI data found for sub-{int(pid):04d}"}]
    items = []
    for session in sorted(os.listdir(sub_dir)):
        sess_path = os.path.join(sub_dir, session)
        if not os.path.isdir(sess_path):
            continue
        items.append({"type": "text", "text": f"\n=== {session} ==="})
        for fname in sorted(os.listdir(sess_path)):
            fpath = os.path.join(sess_path, fname)
            if fname.endswith(".txt"):
                items.append({"type": "text", "text": f"\n{fname}:\n{load_text(fpath)}"})
            elif fname.endswith(".png"):
                items.append({"type": "image", "image": Image.open(fpath)})
                items.append({"type": "text", "text": f"[Image: {fname}]"})
    return items or [{"type": "text", "text": "No MRI data found"}]


def build_prompt(text, has_mri):
    preamble = (
        "You are given patient clinical information and their MRI data "
        "(brain parcellation volume, visualization of brain regions)."
        if has_mri
        else "You are given patient clinical information."
    )
    return (
        preamble + "\n"
        "Classify the patient into one of the following categories:\n"
        "- Major Depressive Disorder\n"
        "- Control (no disorder detected)\n\n"
        "Return your answer as a JSON object with two fields:\n"
        '- "category": the chosen category (exactly one of the two above)\n'
        '- "explanation": a short reasoning for the choice\n\n'
        f"Patient data:\n{text}\n\n"
    )


def find_category_token_idx(gen_tokens, gen_ids=None, t_major=None, t_control=None):
    cat_key_idx = next(
        (i for i, tok in enumerate(gen_tokens) if "category" in tok.lower()),
        None,
    )
    if cat_key_idx is None:
        return None
    if gen_ids is not None and t_major is not None and t_control is not None:
        for i in range(cat_key_idx + 1, len(gen_ids)):
            if gen_ids[i] in (t_major, t_control):
                return i
    for i in range(cat_key_idx + 1, len(gen_tokens)):
        if gen_tokens[i].strip().strip('"') in ("Major", "Control"):
            return i
    return None


def compute_joint_probabilities(scores, cat_idx, t_major, t_control):
    log_p_prefix = 0.0
    for k in range(cat_idx):
        natural = int(torch.argmax(scores[k][0]).item())
        log_p_prefix += torch.log_softmax(scores[k][0], dim=-1)[natural].item()

    label_probs = torch.softmax(scores[cat_idx][0], dim=-1)
    p_major = float(label_probs[t_major].item())
    p_ctrl = float(label_probs[t_control].item())

    eps = 1e-30
    log_p_joint_mdd = log_p_prefix + float(np.log(p_major + eps))
    log_p_joint_ctrl = log_p_prefix + float(np.log(p_ctrl + eps))

    m = max(log_p_joint_mdd, log_p_joint_ctrl)
    a = float(np.exp(log_p_joint_mdd - m))
    b = float(np.exp(log_p_joint_ctrl - m))

    return {
        "log_p_prefix": float(log_p_prefix),
        "p_major_at_label": p_major,
        "p_control_at_label": p_ctrl,
        "log_p_joint_mdd": float(log_p_joint_mdd),
        "log_p_joint_ctrl": float(log_p_joint_ctrl),
        "p_mdd_norm": a / (a + b),
        "p_ctrl_norm": b / (a + b),
        "decision_margin": p_major - p_ctrl,
    }


def _null_record(pred):
    return {
        "log_p_prefix": 0.0,
        "p_major_at_label": 0.5,
        "p_control_at_label": 0.5,
        "log_p_joint_mdd": 0.0,
        "log_p_joint_ctrl": 0.0,
        "p_mdd_norm": 0.5,
        "p_ctrl_norm": 0.5,
        "decision_margin": 0.0,
        "pred_label": pred,
        "cat_idx": None,
        "prefix_length": None,
        "cat_idx_found": False,
    }


def encode_label_in_json_context(tokenizer_encode_fn, label):
    # tokenize the label inside `{"category": "<LABEL>", "x": "y"}` then diff
    # against the same template with a placeholder to recover the label ids
    # without surrounding punctuation/whitespace tokenization noise.
    placeholder = "PLACEHOLDER"
    ids_with = tokenizer_encode_fn(f'OUTPUT: {{"category": "{label}", "x": "y"}}')
    ids_without = tokenizer_encode_fn(f'OUTPUT: {{"category": "{placeholder}", "x": "y"}}')

    n = min(len(ids_with), len(ids_without))
    diverge = next((i for i in range(n) if ids_with[i] != ids_without[i]), None)
    if diverge is None:
        return tokenizer_encode_fn(label)
    suffix = ids_without[diverge:]
    for end in range(diverge, len(ids_with)):
        if ids_with[end:end + len(suffix)] == suffix:
            return ids_with[diverge:end]
    diff_len = len(ids_with) - len(ids_without)
    return ids_with[diverge:diverge + max(1, diff_len)]


class QwenJointHandler:
    def __init__(self, config):
        self.config = config
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.tokenizer = self.processor.tokenizer

        mdd_ids = self.tokenizer.encode("Major Depressive Disorder", add_special_tokens=False)
        ctrl_ids = self.tokenizer.encode("Control", add_special_tokens=False)
        self.t_major = mdd_ids[0]
        self.t_control = ctrl_ids[0]

    def _build_messages(self, text, mri_content, force_preamble):
        prompt = build_prompt(text, has_mri=bool(mri_content) or force_preamble)
        user_content = [{"type": "text", "text": prompt}]
        if mri_content:
            user_content += mri_content
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]

    def classify_joint(self, text, mri_content, force_preamble):
        messages = self._build_messages(text, mri_content, force_preamble)
        tmpl = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        img_in, vid_in = process_vision_info(messages)
        inputs = self.processor(
            text=[tmpl], images=img_in, videos=vid_in, padding=True, return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        inp_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )

        gen_ids = gen.sequences[0][inp_len:]
        id_list = gen_ids.tolist()
        gen_tokens = [self.tokenizer.decode(t) for t in gen_ids]
        cat_idx = find_category_token_idx(gen_tokens, id_list, self.t_major, self.t_control)
        if cat_idx is None:
            decoded = self.processor.decode(gen_ids, skip_special_tokens=True)
            return _null_record("mdd" if "Major" in decoded else "control")

        probs = compute_joint_probabilities(gen.scores, cat_idx, self.t_major, self.t_control)
        probs["pred_label"] = "mdd" if probs["p_mdd_norm"] > probs["p_ctrl_norm"] else "control"
        probs["cat_idx"] = int(cat_idx)
        probs["prefix_length"] = int(cat_idx)
        probs["cat_idx_found"] = True
        return probs


class MistralJointHandler:
    def __init__(self, config):
        self.config = config
        try:
            from transformers import (
                Mistral3ForConditionalGeneration,
                MistralCommonBackend,
            )
        except ImportError:
            raise ImportError("need transformers with Mistral3 support")

        self.processor = MistralCommonBackend.from_pretrained(config.model_name)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            config.model_name, device_map="auto"
        )
        self.tokenizer = self.processor

        def encode(s):
            messages = [{"role": "user", "content": [{"type": "text", "text": s}]}]
            return self.processor.apply_chat_template(
                messages, return_tensors="pt", return_dict=True
            )["input_ids"][0].tolist()

        self.t_major = encode_label_in_json_context(encode, "Major Depressive Disorder")[0]
        self.t_control = encode_label_in_json_context(encode, "Control")[0]

    @staticmethod
    def _image_data_url(image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _build_inputs(self, text, mri_content, force_preamble):
        prompt = build_prompt(text, has_mri=bool(mri_content) or force_preamble)
        user_content = [{"type": "text", "text": prompt}]
        if mri_content:
            for item in mri_content:
                if item["type"] == "text":
                    user_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": self._image_data_url(item["image"])},
                    })
        messages = [{"role": "user", "content": user_content}]
        tokenized = self.processor.apply_chat_template(
            messages, return_tensors="pt", return_dict=True
        )
        for k, v in tokenized.items():
            if isinstance(v, torch.Tensor):
                tokenized[k] = v.to(self.model.device)
        if "pixel_values" in tokenized:
            tokenized["pixel_values"] = tokenized["pixel_values"].to(dtype=torch.bfloat16)
        return tokenized

    @staticmethod
    def _image_sizes(tokenized):
        if "pixel_values" not in tokenized:
            return None
        h, w = tokenized["pixel_values"].shape[-2:]
        return [(h, w)] * tokenized["pixel_values"].shape[0]

    def classify_joint(self, text, mri_content, force_preamble):
        inputs = self._build_inputs(text, mri_content, force_preamble)
        inp_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            gen = self.model.generate(
                **inputs,
                image_sizes=self._image_sizes(inputs),
                max_new_tokens=self.config.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )

        gen_ids = gen.sequences[0][inp_len:]
        id_list = gen_ids.tolist()
        gen_tokens = [self.tokenizer.decode([t]) for t in id_list]
        cat_idx = find_category_token_idx(gen_tokens, id_list, self.t_major, self.t_control)
        if cat_idx is None:
            decoded = self.tokenizer.decode(id_list)
            return _null_record("mdd" if "Major" in decoded else "control")

        probs = compute_joint_probabilities(gen.scores, cat_idx, self.t_major, self.t_control)
        probs["pred_label"] = "mdd" if probs["p_mdd_norm"] > probs["p_ctrl_norm"] else "control"
        probs["cat_idx"] = int(cat_idx)
        probs["prefix_length"] = int(cat_idx)
        probs["cat_idx_found"] = True
        return probs


class InternVLJointHandler:
    # InternVL's chat() doesn't return generate() output, so monkey-patch
    # model.generate for one call to capture sequences + scores. chat() still
    # gets a Tensor (via .sequences) so its downstream decode works.

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, config):
        from transformers import AutoModel
        self.config = config
        self.model = AutoModel.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True, use_fast=False
        )
        # InternVL3.5 shares Qwen2.5's tokenizer base, so direct encoding
        # generally matches what the model emits inside a JSON quote.
        self.t_major = self.tokenizer.encode("Major Depressive Disorder", add_special_tokens=False)[0]
        self.t_control = self.tokenizer.encode("Control", add_special_tokens=False)[0]

    @staticmethod
    def _build_transform(input_size):
        m, s = InternVLJointHandler.IMAGENET_MEAN, InternVLJointHandler.IMAGENET_STD
        return T.Compose([
            T.Lambda(lambda im: im.convert("RGB") if im.mode != "RGB" else im),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=m, std=s),
        ])

    @staticmethod
    def _closest_aspect_ratio(ar, ratios, w, h, sz):
        best_diff, best = float("inf"), (1, 1)
        area = w * h
        for r in ratios:
            d = abs(ar - r[0] / r[1])
            if d < best_diff:
                best_diff, best = d, r
            elif d == best_diff and area > 0.5 * sz * sz * r[0] * r[1]:
                best = r
        return best

    @staticmethod
    def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
        ow, oh = image.size
        ar = ow / oh
        ratios = sorted(
            {(i, j) for n in range(min_num, max_num + 1)
             for i in range(1, n + 1) for j in range(1, n + 1)
             if min_num <= i * j <= max_num},
            key=lambda x: x[0] * x[1],
        )
        tr = InternVLJointHandler._closest_aspect_ratio(ar, ratios, ow, oh, image_size)
        tw, th = image_size * tr[0], image_size * tr[1]
        blocks = tr[0] * tr[1]
        resized = image.resize((tw, th))
        cols = tw // image_size
        crops = [resized.crop((
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )) for i in range(blocks)]
        if use_thumbnail and len(crops) != 1:
            crops.append(image.resize((image_size, image_size)))
        return crops

    def _process_image(self, image, input_size=448, max_num=12):
        tx = self._build_transform(input_size)
        crops = self._dynamic_preprocess(
            image.convert("RGB"),
            image_size=input_size,
            use_thumbnail=True,
            max_num=max_num,
        )
        return torch.stack([tx(c) for c in crops])

    def _assemble(self, text, mri_content, force_preamble):
        prompt = build_prompt(text, has_mri=bool(mri_content) or force_preamble)
        mri_text = ""
        pixel_values = None
        num_patches_list = None
        if mri_content:
            pv_list, npl, idx = [], [], 1
            for item in mri_content:
                if item["type"] == "text":
                    mri_text += item["text"] + "\n"
                elif item["type"] == "image":
                    mri_text += f"Image-{idx}: <image>\n"
                    pv = self._process_image(item["image"])
                    pv_list.append(pv)
                    npl.append(pv.size(0))
                    idx += 1
            if pv_list:
                pixel_values = torch.cat(pv_list, dim=0).to(self.model.device, dtype=torch.bfloat16)
                num_patches_list = npl
        question = prompt + ("\n" + mri_text if mri_text else "")
        return question, pixel_values, num_patches_list

    def classify_joint(self, text, mri_content, force_preamble):
        question, pixel_values, num_patches_list = self._assemble(text, mri_content, force_preamble)

        captured = {}
        original_generate = self.model.generate

        def patched(*args, **kwargs):
            kwargs["return_dict_in_generate"] = True
            kwargs["output_scores"] = True
            out = original_generate(*args, **kwargs)
            captured["out"] = out
            return out.sequences

        self.model.generate = patched
        try:
            with torch.inference_mode():
                self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config={
                        "max_new_tokens": self.config.max_new_tokens,
                        "do_sample": False,
                    },
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )
        finally:
            self.model.generate = original_generate

        gen = captured["out"]
        # input_len = total length minus number of scored (new) tokens.
        input_len = gen.sequences.shape[-1] - len(gen.scores)
        gen_ids = gen.sequences[0][input_len:]
        id_list = gen_ids.tolist()
        gen_tokens = [self.tokenizer.decode([t]) for t in id_list]
        cat_idx = find_category_token_idx(gen_tokens, id_list, self.t_major, self.t_control)
        if cat_idx is None:
            decoded = self.tokenizer.decode(id_list)
            return _null_record("mdd" if "Major" in decoded else "control")

        probs = compute_joint_probabilities(gen.scores, cat_idx, self.t_major, self.t_control)
        probs["pred_label"] = "mdd" if probs["p_mdd_norm"] > probs["p_ctrl_norm"] else "control"
        probs["cat_idx"] = int(cat_idx)
        probs["prefix_length"] = int(cat_idx)
        probs["cat_idx_found"] = True
        return probs


class GLMJointHandler:
    def __init__(self, config):
        from transformers import Glm4vForConditionalGeneration
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval().cuda()
        self.tokenizer = self.processor.tokenizer

        encode = lambda s: self.tokenizer.encode(s, add_special_tokens=False)
        self.t_major = encode_label_in_json_context(encode, "Major Depressive Disorder")[0]
        self.t_control = encode_label_in_json_context(encode, "Control")[0]

    def _build_messages(self, text, mri_content, force_preamble):
        prompt = build_prompt(text, has_mri=bool(mri_content) or force_preamble)
        user_content = [{"type": "text", "text": prompt}]
        if mri_content:
            for item in mri_content:
                if item["type"] == "text":
                    user_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    user_content.append({"type": "image", "image": item["image"]})
        return [{"role": "user", "content": user_content}]

    def classify_joint(self, text, mri_content, force_preamble):
        messages = self._build_messages(text, mri_content, force_preamble)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

        inp_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )

        gen_ids = gen.sequences[0][inp_len:]
        id_list = gen_ids.tolist()
        gen_tokens = [self.tokenizer.decode([t]) for t in id_list]

        think_end = next((i for i, t in enumerate(gen_tokens) if "</think>" in t), None)
        post_tokens = gen_tokens[think_end + 1:] if think_end is not None else gen_tokens
        post_ids = id_list[think_end + 1:] if think_end is not None else id_list

        cat_idx_in_post = find_category_token_idx(post_tokens, post_ids, self.t_major, self.t_control)
        if cat_idx_in_post is None:
            decoded = self.tokenizer.decode(id_list)
            pred = "mdd" if '"M' in decoded or '"Major' in decoded else "control"
            return _null_record(pred)

        offset = (think_end + 1) if think_end is not None else 0
        cat_idx = offset + cat_idx_in_post
        probs = compute_joint_probabilities(gen.scores, cat_idx, self.t_major, self.t_control)
        probs["pred_label"] = "mdd" if probs["p_mdd_norm"] > probs["p_ctrl_norm"] else "control"
        probs["cat_idx"] = int(cat_idx)
        probs["prefix_length"] = int(cat_idx)
        probs["cat_idx_found"] = True
        return probs


def make_handler(config):
    name = config.model_name.lower()
    if "ministral" in name or ("mistral" in name and "2512" in name):
        return MistralJointHandler(config)
    if "qwen2.5-vl" in name or "qwen2_5_vl" in name:
        return QwenJointHandler(config)
    if "internvl" in name:
        return InternVLJointHandler(config)
    if "glm" in name and "4.1v" in name:
        return GLMJointHandler(config)
    raise ValueError(f"Unsupported model: {config.model_name}")


# TODO: C3 (image only) and C5 (OOD swap) not handled here -- C1/C2/C4 only
CONDITION_FLAGS = {
    "c1": (False, False),
    "c2": (False, True),
    "c4": (True, True),
}


def run(handler, config):
    use_mri, force_preamble = CONDITION_FLAGS[config.condition]
    true_label = config.true_label
    n_done = n_correct = 0

    with open(config.output_file, "w", encoding="utf-8") as out_f:
        for filename in tqdm(sorted(
            f for f in os.listdir(config.txt_path) if f.lower().endswith(".txt")
        )):
            txt_path = os.path.join(config.txt_path, filename)
            text = load_text(txt_path)
            mri = (
                get_mri_content(filename, config.mri_path)
                if use_mri and config.mri_path
                else None
            )
            probs = handler.classify_joint(text, mri, force_preamble)
            correct = int(probs["pred_label"] == true_label)

            record = {
                "filename": filename,
                "true_label": true_label,
                "condition": config.condition,
                "correct": correct,
                **probs,
                "timestamp": datetime.now().isoformat(),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_done += 1
            n_correct += correct


def parse_args():
    p = argparse.ArgumentParser(description="Joint-probability inference across C1/C2/C4 (FOR2107 MDD).")
    p.add_argument("--txt_base_path", required=True)
    p.add_argument("--true_label", required=True, choices=["mdd", "control"])
    p.add_argument("--condition", required=True, choices=["c1", "c2", "c4"])
    p.add_argument("--mri_base_path", default=None)
    p.add_argument("--model_name", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.add_argument("--output_dir", default=".")
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=666)
    return p.parse_args()


def main():
    args = parse_args()
    if args.condition == "c4" and not args.mri_base_path:
        raise ValueError("condition c4 needs --mri_base_path")
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    model_base = args.model_name.split("/")[-1].replace("-", "_")

    for split in args.splits:
        txt_path = os.path.join(args.txt_base_path, split)
        mri_path = os.path.join(args.mri_base_path, split) if args.mri_base_path else None
        if not os.path.exists(txt_path):
            print(f"[skip] not found: {txt_path}")
            continue
        out_file = os.path.join(
            args.output_dir,
            f"joint_{model_base}_{args.condition}_{args.true_label}_{split}.jsonl",
        )
        config = argparse.Namespace(
            txt_path=txt_path,
            mri_path=mri_path,
            true_label=args.true_label,
            condition=args.condition,
            output_file=out_file,
            model_name=args.model_name,
            max_new_tokens=args.max_new_tokens,
        )
        run(make_handler(config), config)


if __name__ == "__main__":
    main()

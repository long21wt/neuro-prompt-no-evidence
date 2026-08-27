import argparse
import base64
import io
import json
import os
from datetime import datetime

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Glm4vForConditionalGeneration,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    set_seed,
)

SYSTEM_PROMPT = "You are a helpful medical assistant in clinical psychiatry."


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_mri_content(txt_filename, mri_base_path, include_images=False):
    patient_id = os.path.splitext(txt_filename)[0]
    sub_folder = f"sub-{int(patient_id):04d}"
    subject_path = os.path.join(mri_base_path, sub_folder)
    if not os.path.exists(subject_path):
        return [{"type": "text", "text": f"No MRI data found for {sub_folder}"}]

    items = []
    for session in sorted(os.listdir(subject_path)):
        session_path = os.path.join(subject_path, session)
        if not os.path.isdir(session_path):
            continue
        items.append({"type": "text", "text": f"\n=== {session} ==="})
        for fname in sorted(os.listdir(session_path)):
            fpath = os.path.join(session_path, fname)
            if fname.endswith(".txt"):
                items.append({"type": "text", "text": f"\n{fname}:\n{load_text(fpath)}"})
            elif fname.endswith(".png") and include_images:
                items.append({"type": "image", "image": Image.open(fpath)})
                items.append({"type": "text", "text": f"[Image: {fname}]"})
    return items or [{"type": "text", "text": "No MRI data found"}]


def get_mri_images_only(txt_filename, mri_base_path):
    patient_id = os.path.splitext(txt_filename)[0]
    sub_folder = f"sub-{int(patient_id):04d}"
    subject_path = os.path.join(mri_base_path, sub_folder)
    if not os.path.exists(subject_path):
        return [{"type": "text", "text": f"No MRI data found for {sub_folder}"}]

    items = []
    for session in sorted(os.listdir(subject_path)):
        session_path = os.path.join(subject_path, session)
        if not os.path.isdir(session_path):
            continue
        items.append({"type": "text", "text": f"\n=== {session} ==="})
        for fname in sorted(os.listdir(session_path)):
            if fname.endswith(".png"):
                items.append({"type": "image", "image": Image.open(os.path.join(session_path, fname))})
                items.append({"type": "text", "text": f"[Image: {fname}]"})
    return items or [{"type": "text", "text": "No MRI images found"}]


def get_mri_text_only(txt_filename, mri_base_path):
    items = get_mri_content(txt_filename, mri_base_path, include_images=False)
    parts = [i["text"] for i in items if i["type"] == "text"]
    return "\n".join(parts) if parts else "No MRI text data found"


def build_prompt(text, mri_data=None, include_patient_data=True):
    parts = ["You are given patient"]
    if include_patient_data:
        # C1 preamble. mri_data appends the MRI mention below (C2/C4). For the
        # false-modality ablation (Section 7), swap in "clinical information
        # and fMRI data" or "...and The weather is sunny today".
        # TODO: C5 (swap-image OOD) is not exposed here; do it in a notebook
        #       or patch inference_joint when we get to it.
        # parts.append("clinical information and Magnetic Resonance Imaging data")
        parts.append("clinical information")
    if mri_data:
        parts.append("and their MRI data (brain parcellation volume, visualization of brain regions)")

    prompt = " ".join(parts) + ".\n"
    prompt += (
        "Classify the patient into one of the following categories:\n"
        "- Major Depressive Disorder\n"
        "- Control (no disorder detected)\n\n"
        "Return your answer as a JSON object with two fields:\n"
        '- "category": the chosen category (exactly one of the two above)\n'
        '- "explanation": a short reasoning for the choice\n\n'
    )
    if include_patient_data:
        prompt += f"Patient data:\n{text}\n\n"
    if mri_data and isinstance(mri_data, str) and mri_data != "multimodal":
        prompt += "Patient MRI parcellation data:\n" + mri_data
    return prompt


class Handler:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        raise NotImplementedError

    def classify(self, text, mri_content=None):
        raise NotImplementedError


class InternVLHandler(Handler):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def _load_model(self):
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True, use_fast=False
        )

    @staticmethod
    def _build_transform(input_size):
        mean, std = InternVLHandler.IMAGENET_MEAN, InternVLHandler.IMAGENET_STD
        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    @staticmethod
    def _closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_diff = float("inf")
        best = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target = ratio[0] / ratio[1]
            diff = abs(aspect_ratio - target)
            if diff < best_diff:
                best_diff = diff
                best = ratio
            elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best = ratio
        return best

    @staticmethod
    def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_w, orig_h = image.size
        aspect_ratio = orig_w / orig_h
        target_ratios = sorted(
            {(i, j) for n in range(min_num, max_num + 1)
             for i in range(1, n + 1) for j in range(1, n + 1)
             if min_num <= i * j <= max_num},
            key=lambda x: x[0] * x[1],
        )
        target = InternVLHandler._closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_w, orig_h, image_size
        )
        tw, th = image_size * target[0], image_size * target[1]
        blocks = target[0] * target[1]
        resized = image.resize((tw, th))
        out = []
        for i in range(blocks):
            box = (
                (i % (tw // image_size)) * image_size,
                (i // (tw // image_size)) * image_size,
                ((i % (tw // image_size)) + 1) * image_size,
                ((i // (tw // image_size)) + 1) * image_size,
            )
            out.append(resized.crop(box))
        if use_thumbnail and len(out) != 1:
            out.append(image.resize((image_size, image_size)))
        return out

    def _process_image(self, image_obj, input_size=448, max_num=12):
        image = image_obj.convert("RGB")
        transform = self._build_transform(input_size=input_size)
        images = self._dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        return torch.stack([transform(img) for img in images])

    def classify(self, text, mri_content=None):
        pixel_values = None
        num_patches_list = None
        mri_text = ""

        if isinstance(mri_content, str):
            mri_text = mri_content
        elif isinstance(mri_content, list):
            pv_list = []
            num_patches_list = []
            idx = 1
            for item in mri_content:
                if item["type"] == "text":
                    mri_text += item["text"] + "\n"
                elif item["type"] == "image":
                    mri_text += f"Image-{idx}: <image>\n"
                    pv = self._process_image(item["image"])
                    pv_list.append(pv)
                    num_patches_list.append(pv.size(0))
                    idx += 1
            if pv_list:
                pixel_values = torch.cat(pv_list, dim=0).to(self.model.device, dtype=torch.bfloat16)

        if pixel_values is not None:
            base = build_prompt(text, "multimodal", include_patient_data=bool(text))
            question = base + "\n" + mri_text
        else:
            mri_arg = mri_text if mri_text.strip() else None
            question = build_prompt(text, mri_arg, include_patient_data=bool(text))

        response, _ = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            dict(max_new_tokens=self.config.max_new_tokens, do_sample=self.config.do_sample),
            num_patches_list=num_patches_list,
            history=None,
            return_history=True,
        )
        return response.strip()


class MinistralHandler(Handler):
    def _load_model(self):
        try:
            from transformers import (
                Mistral3ForConditionalGeneration,
                MistralCommonBackend,
            )
        except ImportError:
            raise ImportError("need transformers with Mistral3 support")
        self.processor = MistralCommonBackend.from_pretrained(self.config.model_name)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            self.config.model_name, device_map="auto"
        )

    @staticmethod
    def _image_data_url(image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def classify(self, text, mri_content=None):
        prompt = build_prompt(
            text,
            "multimodal" if isinstance(mri_content, list) else mri_content,
            include_patient_data=bool(text),
        )
        user_content = [{"type": "text", "text": prompt}]
        if isinstance(mri_content, list):
            for item in mri_content:
                if item["type"] == "text":
                    user_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    # Data URI works around strict schema validation in MistralCommonBackend.
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
        image_sizes = None
        if "pixel_values" in tokenized:
            tokenized["pixel_values"] = tokenized["pixel_values"].to(
                dtype=torch.bfloat16, device=self.model.device
            )
            h, w = tokenized["pixel_values"].shape[-2:]
            image_sizes = [(h, w) for _ in range(tokenized["pixel_values"].shape[0])]

        with torch.inference_mode():
            out = self.model.generate(
                **tokenized,
                image_sizes=image_sizes,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )[0]
        return self.processor.decode(out[len(tokenized["input_ids"][0]):]).strip()


class PixtralHandler(Handler):
    def _load_model(self):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.config.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)

    def classify(self, text, mri_content=None):
        prompt = build_prompt(text,
                              "multimodal" if isinstance(mri_content, list) else mri_content,
                              include_patient_data=bool(text))
        content = [{"type": "text", "text": prompt}]
        pil_images = []
        if isinstance(mri_content, list):
            for item in mri_content:
                if item["type"] == "text":
                    content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    content.append({"type": "image"})
                    pil_images.append(item["image"])
        conv = [{"role": "user", "content": content}]
        chat_prompt = self.processor.apply_chat_template(conv, add_generation_prompt=True)
        if pil_images:
            inputs = self.processor(text=chat_prompt, images=pil_images, return_tensors="pt")
        else:
            inputs = self.processor(text=chat_prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            gen_out = self.model.generate(**inputs,
                                          max_new_tokens=self.config.max_new_tokens,
                                          do_sample=self.config.do_sample)
        return self.processor.decode(gen_out[0][input_len:], skip_special_tokens=True).strip()


class GemmaHandler(Handler):
    def _load_model(self):
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_name, dtype=torch.bfloat16,
            device_map="auto", attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)

    def classify(self, text, mri_content=None):
        if isinstance(mri_content, list):
            prompt = build_prompt(text, "multimodal", include_patient_data=bool(text))
            user_content = [{"type": "text", "text": prompt}] + list(mri_content)
        else:
            prompt = build_prompt(text, mri_content, include_patient_data=bool(text))
            user_content = [{"type": "text", "text": prompt}]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        in_len = inputs.input_ids.shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )
        return self.processor.decode(generation[0][in_len:], skip_special_tokens=True).strip()


class Qwen2VLHandler(Handler):
    _MODEL_CLS = Qwen2VLForConditionalGeneration

    def _load_model(self):
        from qwen_vl_utils import process_vision_info
        self._process_vision_info = process_vision_info
        self.model = self._MODEL_CLS.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)

    def classify(self, text, mri_content=None):
        if isinstance(mri_content, list):
            prompt = build_prompt(text, "multimodal", include_patient_data=bool(text))
            user_content = [{"type": "text", "text": prompt}] + list(mri_content)
        else:
            prompt = build_prompt(text, mri_content, include_patient_data=bool(text))
            user_content = [{"type": "text", "text": prompt}]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]
        text_in = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(text=[text_in], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt"
                                ).to(self.model.device, dtype=torch.bfloat16)
        ilen = inputs.input_ids.shape[-1]
        with torch.inference_mode():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )
        return self.processor.decode(gen_ids[0][ilen:], skip_special_tokens=True).strip()


class Qwen2_5VLHandler(Qwen2VLHandler):
    _MODEL_CLS = Qwen2_5_VLForConditionalGeneration


class Qwen2_5Handler(Handler):
    # text-only Qwen2.5, kept around for the C1 baseline on non-VL setups
    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name, dtype=torch.bfloat16,
                                                          device_map="auto",
                                                          attn_implementation="flash_attention_2")
        self.processor = AutoTokenizer.from_pretrained(self.config.model_name)

    def classify(self, text, mri_content=None):
        if isinstance(mri_content, list):
            raise ValueError("qwen2.5 text-only, mri_content can't be a list")
        prompt = build_prompt(text, mri_content, include_patient_data=bool(text))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        chat = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[chat], padding=True, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens,
                                      do_sample=self.config.do_sample)
        return self.processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


class Qwen3VLHandler(Handler):
    _MODEL_CLS = Qwen3VLForConditionalGeneration

    def _load_model(self):
        self.model = self._MODEL_CLS.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)

    def classify(self, text, mri_content=None):
        if isinstance(mri_content, list):
            prompt = build_prompt(text, "multimodal", include_patient_data=bool(text))
            user_content = [{"type": "text", "text": prompt}] + list(mri_content)
        else:
            prompt = build_prompt(text, mri_content, include_patient_data=bool(text))
            user_content = [{"type": "text", "text": prompt}]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]
        inp = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = inp.input_ids.shape[-1]
        with torch.inference_mode():
            output = self.model.generate(**inp,
                                         max_new_tokens=self.config.max_new_tokens,
                                         do_sample=self.config.do_sample)
        return self.processor.decode(output[0][input_len:], skip_special_tokens=True).strip()


class Qwen3VLMoeHandler(Qwen3VLHandler):
    _MODEL_CLS = Qwen3VLMoeForConditionalGeneration


class LlavaOneVisionHandler(Handler):
    def _load_model(self):
        from qwen_vl_utils import process_vision_info
        self._process_vision_info = process_vision_info
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            force_download=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )

    def classify(self, text, mri_content=None):
        # llava-onevision: no system role in chat template
        if isinstance(mri_content, list):
            prompt = build_prompt(text, "multimodal", include_patient_data=bool(text))
            user_content = [{"type": "text", "text": prompt}] + list(mri_content)
        else:
            prompt = build_prompt(text, mri_content, include_patient_data=bool(text))
            user_content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": user_content}]
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[chat],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs,
                                      max_new_tokens=self.config.max_new_tokens,
                                      do_sample=self.config.do_sample)
        trimmed = [g[len(inp):] for inp, g in zip(inputs.input_ids, out)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]


class Glm4vHandler(Handler):
    def _load_model(self):
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.config.model_name, use_fast=True)

    def classify(self, text, mri_content=None):
        # GLM-4V wants images/text dicts BEFORE the trailing prompt string
        prompt = build_prompt(text,
                              "multimodal" if isinstance(mri_content, list) else mri_content,
                              include_patient_data=bool(text))
        uc = []
        if isinstance(mri_content, list):
            for item in mri_content:
                if item["type"] == "text":
                    uc.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    uc.append({"type": "image", "image": item["image"]})
        uc.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": uc}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)
                if k in ("images", "pixel_values"):
                    inputs[k] = inputs[k].to(dtype=torch.bfloat16)
        ilen = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            gen = self.model.generate(**inputs,
                                      max_new_tokens=self.config.max_new_tokens,
                                      do_sample=self.config.do_sample)
        return self.processor.decode(gen[0][ilen:], skip_special_tokens=True).strip()


def make_handler(config):
    name = config.model_name.lower()
    if "internvl" in name:
        return InternVLHandler(config)
    if "ministral" in name or ("mistral" in name and "2512" in name):
        return MinistralHandler(config)
    if "glm" in name:
        return Glm4vHandler(config)
    if "llava-onevision" in name:
        return LlavaOneVisionHandler(config)
    if "pixtral" in name:
        return PixtralHandler(config)
    if "gemma" in name or "medgemma" in name:
        return GemmaHandler(config)
    if "qwen2-vl" in name:
        return Qwen2VLHandler(config)
    if "qwen2.5-vl" in name or "qwen2_5_vl" in name:
        return Qwen2_5VLHandler(config)
    if "qwen3-vl" in name and ("moe" in name or "a3b" in name or "a22b" in name):
        return Qwen3VLMoeHandler(config)
    if "qwen3-vl" in name:
        return Qwen3VLHandler(config)
    if "qwen2.5" in name or "qwen2_5" in name:
        return Qwen2_5Handler(config)
    raise ValueError(f"unsupported model: {config.model_name}")


def run(handler, config, mode):
    mri = config.mri_base_path
    with open(config.output_file, "w", encoding="utf-8") as out_f:
        for filename in tqdm(os.listdir(config.txt_path)):
            if not filename.lower().endswith(".txt"):
                continue
            file_path = os.path.join(config.txt_path, filename)
            patient_data = load_text(file_path)

            mri_content = None
            summary = None
            if mode == "tabular":
                pass
            elif mode == "tabular_parcel" and mri:
                mri_content = get_mri_text_only(filename, mri)
                summary = mri_content
            elif mode == "tabular_mri" and mri:
                mri_content = get_mri_images_only(filename, mri)
                # mri_content is a list of {type,text|image} dicts
                summary = "\n".join(
                    i["text"] if i["type"] == "text" else "[Image data included in processing]"
                    for i in mri_content
                )
            elif mode in ("tabular_parcel_mri", "parcel_mri") and mri:
                mri_content = get_mri_content(filename, mri, include_images=True)
                parts = []
                for i in mri_content:
                    if i["type"] == "text":
                        parts.append(i["text"])
                    else:
                        parts.append("[Image data included in processing]")
                summary = "\n".join(parts)
            elif mode not in ("tabular", "tabular_parcel", "tabular_mri",
                              "tabular_parcel_mri", "parcel_mri"):
                raise ValueError(f"unknown mode: {mode}")

            text_input = "" if mode == "parcel_mri" else patient_data
            category = handler.classify(text_input, mri_content)

            record = {
                "filename": filename,
                "full_path": file_path,
                "input": patient_data,
                "output": category,
                "timestamp": datetime.now().isoformat(),
            }
            if summary is not None:
                record["mri_data_summary"] = summary

            # mri_paths: enumerate session subfolders for downstream JSONL reading
            if mri:
                pid = os.path.splitext(filename)[0]
                sub_folder = f"sub-{int(pid):04d}"
                subject_path = os.path.join(mri, sub_folder)
                if os.path.exists(subject_path):
                    paths = {"subject_folder": subject_path, "sessions": {}}
                    for sess in sorted(os.listdir(subject_path)):
                        sp = os.path.join(subject_path, sess)
                        if not os.path.isdir(sp):
                            continue
                        paths["sessions"][sess] = {
                            "session_path": sp,
                            "files": [os.path.join(sp, f) for f in sorted(os.listdir(sp))],
                        }
                    record["mri_paths"] = paths
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Patient classification inference (FOR2107 / MDD).")
    p.add_argument("--txt_path", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--mode", required=True, choices=[
        "tabular", "tabular_parcel", "tabular_mri", "tabular_parcel_mri", "parcel_mri",
    ])
    p.add_argument("--mri_base_path", default=None)
    p.add_argument("--output_file", default=None)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--seed", type=int, default=666)
    return p.parse_args()


def validate(args):
    if args.mode in ("tabular_parcel", "tabular_mri", "tabular_parcel_mri", "parcel_mri"):
        if not args.mri_base_path:
            raise ValueError(f"{args.mode} needs --mri_base_path")
    if not os.path.exists(args.txt_path):
        raise FileNotFoundError(f"txt_path not found: {args.txt_path}")
    if args.mri_base_path and not os.path.exists(args.mri_base_path):
        raise FileNotFoundError(f"MRI base path does not exist: {args.mri_base_path}")
    if not args.output_file:
        model_base = args.model_name.split("/")[-1].replace("-", "_")
        if args.mri_base_path:
            mri_base = args.mri_base_path.split("/")[-2] + "_" + args.mri_base_path.split("/")[-1]
        else:
            mri_base = "no_mri"
        txt_base = args.txt_path.split("/")[-2] + "_" + args.txt_path.split("/")[-1]
        args.output_file = (
            f"results_{model_base}_mri_{mri_base}_txt_{txt_base}_{args.mode}.jsonl"
        )
    return args


if __name__ == "__main__":
    args = validate(parse_args())
    # print(args)
    set_seed(args.seed)
    run(make_handler(args), args, args.mode)

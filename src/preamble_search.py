import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except ImportError:
    pass

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# 1-indexed; the last layer before the label-routing decision forms.
# Layer 33 is where signal first appears; layer 32 is the input to that step.
PREAMBLE_LAYER = 32

MDD_TOKEN = "Major"
CONTROL_TOKEN = "Control"

TABULAR_PREAMBLE = "You are given patient clinical information."
MRI_PREAMBLE = (
    "You are given patient clinical information and their MRI data "
    "(brain parcellation volume, visualization of brain regions)."
)

CANDIDATE_PHRASES = {
    "MRI / neuroimaging": [
        "You are given patient clinical information and their MRI data.",
        "Brain MRI findings are available.",
        "Neuroimaging data is provided.",
        "fMRI data is included.",
        "MRI scan results are attached.",
        "Brain scans have been performed.",
    ],
    "General clinical / diagnostic": [
        "A clinical diagnosis has been established.",
        "The patient has been evaluated by a specialist.",
        "Diagnostic results are available.",
        "Medical records are provided.",
        "The patient has been assessed for a psychiatric disorder.",
        "Clinical evaluation is complete.",
    ],
    "Authoritative framing": [
        "You are an expert clinical psychiatrist.",
        "You are a specialist in mood disorders.",
        "As a medical professional, review the following.",
        "You have extensive experience in psychiatric diagnosis.",
    ],
    "Pathology / disorder priming": [
        "The patient may have a depressive disorder.",
        "Symptoms of depression have been observed.",
        "The patient presents with mood disturbances.",
        "A psychiatric condition is suspected.",
        "The patient reports persistent low mood.",
        "Mental health concerns have been flagged.",
    ],
    "Neutral / unrelated": [
        "The weather is sunny today.",
        "This is a test of the system.",
        "Please process the following information.",
        "Data is provided below.",
        "Answer the following question carefully.",
        "You are a helpful assistant.",
    ],
    "Structural / format priming": [
        "Return your answer as JSON.",
        "Respond only with a JSON object.",
        'Output: {"category":',
        "The answer is:",
        "Classification result:",
    ],
    "Negation / opposite priming": [
        "The patient is healthy and shows no symptoms.",
        "No psychiatric disorder has been detected.",
        "The patient is a control subject.",
        "All clinical indicators are within normal range.",
    ],
}


class PreambleModel:
    def __init__(self, model_name):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.mdd_tok_id = self.tokenizer.encode(MDD_TOKEN, add_special_tokens=False)[0]
        self.ctrl_tok_id = self.tokenizer.encode(CONTROL_TOKEN, add_special_tokens=False)[0]
        self.lm_head = self.model.lm_head
        self.n_hidden = self.lm_head.weight.shape[1]
        self.final_ln = self.model.model.language_model.norm

    def _prepare(self, messages):
        tmpl = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        img_in, vid_in = process_vision_info(messages)
        return self.processor(
            text=[tmpl],
            images=img_in,
            videos=vid_in,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

    def get_label_hidden_state(self, messages, target_layer=PREAMBLE_LAYER, max_new_tokens=80):
        """Generate; return (h_at_target_layer, label, p_mdd, p_ctrl) at the label step, or None."""
        inputs = self._prepare(messages)
        with torch.inference_mode():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )

        in_len = inputs["input_ids"].shape[1]
        new_ids = gen.sequences[0, in_len:].tolist()

        label_step = None
        found_label = None
        for i, tid in enumerate(new_ids):
            if tid == self.mdd_tok_id:
                label_step, found_label = i, "MDD"
                break
            if tid == self.ctrl_tok_id:
                label_step, found_label = i, "Control"
                break
        if found_label is None:
            return None

        hs_at_step = gen.hidden_states[label_step]
        # hidden_states[step] is a tuple of (n_layers+1) tensors of shape (1,1,hidden):
        # index 0 = embedding output; layer i (1-indexed) maps to index i.
        h = hs_at_step[target_layer][0, 0, :].float().cpu().numpy()

        h_final = hs_at_step[-1][0, 0, :].unsqueeze(0).to(
            device=next(self.final_ln.parameters()).device, dtype=torch.bfloat16
        )
        with torch.inference_mode():
            logits = self.lm_head(self.final_ln(h_final).to(self.lm_head.weight.device)).float()[0]
            probs = torch.softmax(logits, dim=-1)
        return h, found_label, probs[self.mdd_tok_id].item(), probs[self.ctrl_tok_id].item()


def build_messages(text, preamble):
    prompt = (
        preamble + "\n"
        "Classify the patient into one of the following categories:\n"
        "- Major Depressive Disorder\n"
        "- Control (no disorder detected)\n\n"
        "Return your answer as a JSON object with two fields:\n"
        '- "category": the chosen category\n'
        '- "explanation": a short reasoning\n\n'
        f"Patient data:\n{text}\n\n"
    )
    return [
        {"role": "system", "content": [{"type": "text",
         "text": "You are a helpful medical assistant in clinical psychiatry."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def load_patients(txt_path, label, n=None):
    out = []
    for fname in sorted(os.listdir(txt_path)):
        if not fname.lower().endswith(".txt"):
            continue
        with open(os.path.join(txt_path, fname)) as f:
            text = f.read()
        out.append((os.path.splitext(fname)[0], text, label))
        if n and len(out) >= n:
            break
    return out


def extract_preamble_direction(model, patients, output_dir, target_layer=PREAMBLE_LAYER):
    # mean(h_mri) - mean(h_tabular) at target_layer, over patients; normalized.
    h_tabular, h_mri = [], []
    for pid, text, _ in tqdm(patients, desc="Extracting"):
        for preamble, store in [(TABULAR_PREAMBLE, h_tabular), (MRI_PREAMBLE, h_mri)]:
            res = model.get_label_hidden_state(build_messages(text, preamble), target_layer=target_layer)
            if res is None:
                tqdm.write(f"  [skip] {pid}/{preamble[:20]}: label not generated")
                store.append(None)
            else:
                store.append(res[0])

    pairs = [(t, m) for t, m in zip(h_tabular, h_mri) if t is not None and m is not None]
    if not pairs:
        raise RuntimeError("No patient produced a label in both conditions.")

    h_tab_arr = np.stack([p[0] for p in pairs])
    h_mri_arr = np.stack([p[1] for p in pairs])
    direction = h_mri_arr.mean(0) - h_tab_arr.mean(0)
    direction_norm = direction / (np.linalg.norm(direction) + 1e-9)

    np.savez(
        os.path.join(output_dir, "preamble_direction.npz"),
        preamble_dir=direction,
        preamble_dir_norm=direction_norm,
        h_tabular=h_tab_arr,
        h_mri=h_mri_arr,
    )
    return direction_norm


def single_token_search(
    model,
    preamble_dir,
    patients,
    output_dir,
    target_layer=PREAMBLE_LAYER,
    top_k=100,
    vocab_sample=5000,
):
    """Prepend each candidate single token to the tabular prompt and rank by
    cosine of its hidden-state shift to preamble_dir. Single patient for speed."""
    pid, text, _ = patients[0]
    tokenizer = model.tokenizer

    candidates = []
    for tid in range(tokenizer.vocab_size):
        tok = tokenizer.decode([tid])
        if tok.strip() and tok.replace(" ", "").isalpha() and len(tok.strip()) >= 2:
            candidates.append(tid)
        if len(candidates) >= vocab_sample:
            break

    base = model.get_label_hidden_state(build_messages(text, TABULAR_PREAMBLE), target_layer=target_layer)
    if base is None:
        return []
    h_base = base[0]

    results = []
    for tid in tqdm(candidates, desc="Token search"):
        tok_str = tokenizer.decode([tid]).strip()
        augmented = tok_str + ". " + TABULAR_PREAMBLE
        res = model.get_label_hidden_state(build_messages(text, augmented), target_layer=target_layer)
        if res is None:
            continue
        shift = res[0] - h_base
        cos = float(np.dot(shift / (np.linalg.norm(shift) + 1e-9), preamble_dir))
        results.append((cos, res[2], tid, tok_str))

    results.sort(reverse=True)
    top = results[:top_k]
    bottom = results[-top_k:]
    with open(os.path.join(output_dir, "token_search_results.json"), "w") as f:
        json.dump({"top_preamble": top, "anti_preamble": bottom}, f, indent=2)
    return top


def phrase_search(model, preamble_dir, patients, output_dir, target_layer=PREAMBLE_LAYER):
    """Score each CANDIDATE_PHRASES entry by (a) cosine sim of hidden-state shift
    to preamble_dir and (b) actual P(MDD) shift vs the tabular baseline, averaged
    across patients."""
    baselines_h, baselines_pm = {}, {}
    for pid, text, _ in tqdm(patients, desc="Baseline"):
        res = model.get_label_hidden_state(build_messages(text, TABULAR_PREAMBLE), target_layer=target_layer)
        if res is None:
            continue
        baselines_h[pid] = res[0]
        baselines_pm[pid] = res[2]

    all_results = []
    for category, phrases in CANDIDATE_PHRASES.items():
        for phrase in phrases:
            cos_sims, p_mdds, p_shifts = [], [], []
            for pid, text, _ in patients:
                if pid not in baselines_h:
                    continue
                res = model.get_label_hidden_state(build_messages(text, phrase), target_layer=target_layer)
                if res is None:
                    continue
                shift = res[0] - baselines_h[pid]
                cos_sims.append(
                    float(np.dot(shift / (np.linalg.norm(shift) + 1e-9), preamble_dir))
                )
                p_mdds.append(res[2])
                p_shifts.append(res[2] - baselines_pm[pid])
            if not cos_sims:
                continue
            all_results.append({
                "category": category,
                "phrase": phrase,
                "cos_sim_mean": float(np.mean(cos_sims)),
                "cos_sim_std": float(np.std(cos_sims)),
                "p_mdd_mean": float(np.mean(p_mdds)),
                "p_mdd_std": float(np.std(p_mdds)),
                "p_shift_mean": float(np.mean(p_shifts)),
                "p_shift_std": float(np.std(p_shifts)),
                "n": len(cos_sims),
            })

    all_results.sort(key=lambda x: x["cos_sim_mean"], reverse=True)
    with open(os.path.join(output_dir, "phrase_search_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


def plot_phrase_results(results, output_dir):
    if not results:
        return
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    categories = list(by_cat)
    cat_cos = [np.mean([r["cos_sim_mean"] for r in by_cat[c]]) for c in categories]
    cat_shift = [np.mean([r["p_shift_mean"] for r in by_cat[c]]) for c in categories]
    order = np.argsort(cat_cos)[::-1]
    categories = [categories[i] for i in order]
    cat_cos = [cat_cos[i] for i in order]
    cat_shift = [cat_shift[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(categories)))[::-1]
    panels = [
        (axes[0], cat_cos, "Cosine sim to preamble direction",
         "Alignment with preamble direction\n(by phrase category)"),
        (axes[1], cat_shift, "delta P(MDD) vs tabular baseline",
         "P(MDD) shift vs tabular-only\n(by phrase category)"),
    ]
    for ax, vals, xlabel, title in panels:
        ax.barh(range(len(categories)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

    fig.suptitle("Preamble Direction Search (FOR2107)", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "preamble_phrase_results.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_top_phrases_scatter(results, output_dir):
    if not results:
        return
    categories = list({r["category"] for r in results})
    cmap = plt.cm.tab10
    color_for = {c: cmap(i / len(categories)) for i, c in enumerate(categories)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        ax.scatter(r["cos_sim_mean"], r["p_shift_mean"],
                   color=color_for[r["category"]], s=60, alpha=0.8, zorder=3)
        if abs(r["cos_sim_mean"]) > 0.1 or abs(r["p_shift_mean"]) > 0.05:
            ax.annotate(
                r["phrase"][:35],
                (r["cos_sim_mean"], r["p_shift_mean"]),
                fontsize=4.5,
                alpha=0.75,
                xytext=(4, 2),
                textcoords="offset points",
            )
    for cat, color in color_for.items():
        ax.scatter([], [], color=color, label=cat, s=40)
    ax.legend(fontsize=6, framealpha=0.9, loc="upper left")
    ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.5)
    ax.axvline(0, color="grey", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Cosine similarity to preamble direction", fontsize=9)
    ax.set_ylabel("delta P(MDD) vs tabular baseline", fontsize=9)
    ax.set_title(
        "Equivalence class of preamble-activating phrases (FOR2107)\n"
        "Top-right quadrant = same effect as MRI preamble",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "preamble_scatter.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Search for phrases that activate the MRI-preamble direction.")
    p.add_argument("--txt_mdd_path", required=True)
    p.add_argument("--txt_ctrl_path", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--n_patients", type=int, default=15,
                   help="Patients per class for direction extraction.")
    p.add_argument("--preamble_layer", type=int, default=PREAMBLE_LAYER)
    p.add_argument("--output_dir", default="./preamble_results")
    p.add_argument("--load_direction", default=None,
                   help="Path to saved preamble_direction.npz; skips extraction.")
    p.add_argument("--skip_token_search", action="store_true")
    p.add_argument("--vocab_sample", type=int, default=5000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = PreambleModel(args.model_name)
    mdd_pts = load_patients(args.txt_mdd_path, "MDD", args.n_patients)
    ctrl_pts = load_patients(args.txt_ctrl_path, "Control", args.n_patients)
    # MDD patients show the clearest shift, so extract the direction from them.
    direction_patients = mdd_pts

    if args.load_direction:
        preamble_dir = np.load(args.load_direction)["preamble_dir_norm"]
    else:
        preamble_dir = extract_preamble_direction(
            model, direction_patients, args.output_dir, args.preamble_layer
        )

    if not args.skip_token_search:
        single_token_search(
            model, preamble_dir, mdd_pts[:3], args.output_dir,
            target_layer=args.preamble_layer, vocab_sample=args.vocab_sample,
        )

    phrase_results = phrase_search(
        model, preamble_dir, mdd_pts, args.output_dir, args.preamble_layer
    )
    plot_phrase_results(phrase_results, args.output_dir)
    plot_top_phrases_scatter(phrase_results, args.output_dir)

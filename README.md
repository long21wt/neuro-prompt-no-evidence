# Prompts Without Evidence: How Neuroimaging Mentions Shift Clinical Vision-Language Model Predictions

Code for our paper (Accepted to EMNL 2026 Main Conference). 

We evaluate Vision-Language Models (VLMs) on two clinical classification tasks using multimodal patient data (clinical text + structural MRI).

## Citation
Incoming

## Tasks

- **MDD classification** (FOR2107 dataset): Major Depressive Disorder vs. Control
- **Cognitive decline classification** (OASIS dataset): Cognitive Decline vs. Cognitive Normal

## Repository Structure

| File | Description |
|------|-------------|
| `inference.py` | Main inference script for MDD task; supports Gemma-3, LLaVA, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, GLM-4V, InternVL |
| `inference_oasis.py` | Inference script for OASIS cognitive decline task |
| `inference_joint.py` | Joint-probability inference: extracts per-token log-probs and computes renormalized P(MDD) across conditions C1/C2/C4 (Appendix `app:confidence`) |
| `train_dpo.py` | MPO/DPO fine-tuning of Qwen2.5-VL-3B-Instruct on a preference dataset |
| `preamble_search.py` | Mechanistic interpretability: extracts preamble direction from layer-33 hidden states and searches for equivalent trigger phrases |
| `summarize_joint.py` | Aggregates `inference_joint.py` outputs into per-condition metrics (mean P(MDD), ECE, Brier and figures |
| `f1_eval.py` | Computes F1 / precision / recall / accuracy for MDD results |
| `f1_eval_oasis.py` | Same evaluation for OASIS results |

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable GPU (≥24 GB VRAM recommended for 7B+ models).

## Usage

### Inference - MDD (FOR2107)

```bash
python inference.py \
    --txt_path     /path/to/txt_mdd_split/test \
    --mri_base_path /path/to/mri_data \
    --output_file  results_test.jsonl \
    --model_name   Qwen/Qwen2.5-VL-7B-Instruct \
    --mode         tabular_parcel_mri
```

### Inference - OASIS

```bash
python inference_oasis.py \
    --txt_path    /path/to/oasis_txt/test \
    --output_file oasis_results_test.jsonl \
    --model_name  Qwen/Qwen2.5-VL-7B-Instruct \
    --mode        tabular
```

### Evaluation - MDD

```bash
python f1_eval.py \
    --control_file results_control.jsonl \
    --mdd_file     results_mdd.jsonl
```

### Evaluation - OASIS

```bash
python f1_eval_oasis.py \
    --cn_file results_cn.jsonl \
    --cd_file results_cd.jsonl
```

### Joint-probability inference (confidence & calibration)

Run one invocation per (model, condition, true-label) cell. Conditions are
`c1` (text only), `c2` (text + MRI preamble, no image), `c4` (text + parcel + MRI image):

```bash
python inference_joint.py \
    --txt_base_path /path/to/txt_mdd_split \
    --true_label    mdd \
    --condition     c2 \
    --mri_base_path /path/to/mri_data \
    --model_name    Qwen/Qwen2.5-VL-3B-Instruct \
    --output_dir    ./results_joint_v3
```

Aggregate the resulting `.jsonl` files into per-condition tables and figures
(mean P(MDD), ECE, Brier):

```bash
python summarize_joint.py \
    --input_dir  ./results_joint_v3 \
    --output_dir ./summary_joint
```

### Fine-tuning (MPO/DPO)

```bash
python train_dpo.py \
    --dataset_dir ./dpo_dataset \
    --output_dir  ./qwen25vl_finetuned \
    --run_name    qwen25vl_mpo
```


### Preamble direction search

```bash
python preamble_search.py \
    --txt_mdd_path  /path/to/txt_mdd_split/test \
    --txt_ctrl_path /path/to/txt_control_split/test \
    --model_name    Qwen/Qwen2.5-VL-3B-Instruct \
    --n_patients    15 \
    --output_dir    ./preamble_results
```

## Input Data Format

Each patient is represented by a plain-text file (`<patient_id>.txt`) containing structured clinical features. For MRI conditions, a corresponding folder `sub-<NNNN>/` holds session subfolders with parcellation stats (`.txt`) and brain region visualisation plots (`.png`).

## Output Format

All inference scripts produce `.jsonl` files where each line is a JSON object:

```json
{
  "filename": "0042.txt",
  "mode": "tabular_parcel_mri",
  "output": "{ \"category\": \"Major Depressive Disorder\", \"explanation\": \"...\" }"
}
```

`inference_joint.py` outputs additionally include per-token confidence fields:
`log_p_prefix`, `p_major_at_label`, `p_control_at_label`, `log_p_joint_mdd`,
`log_p_joint_ctrl`, `p_mdd_norm`, `pred_label`, `cat_idx` (see Appendix
`app:confidence` of the paper).

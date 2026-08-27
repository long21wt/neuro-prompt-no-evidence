import argparse
import os

import torch
from datasets import enable_progress_bars, load_from_disk
from peft import LoraConfig, TaskType
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import DPOConfig, DPOTrainer

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def build_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def load_model_and_processor(model_name):
    # no device_map="auto": multi-GPU DDP needs one process per GPU.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model.config.pad_token_id = processor.tokenizer.eos_token_id
    return model, processor


def load_dataset(dataset_dir, val_split=0.05, seed=42):
    ds = load_from_disk(dataset_dir).shuffle(seed=seed)
    split = ds.train_test_split(test_size=val_split, seed=seed)
    return split["train"], split["test"]


def build_dpo_config(args):
    return DPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        # max_length=None is required for VLMs; truncation clips <|image_pad|>
        # tokens and breaks pixel/text alignment.
        # https://huggingface.co/docs/trl/dpo_trainer#vision-language-models
        max_length=None,
        loss_type=["sigmoid", "bco_pair", "sft"],
        loss_weights=[0.8, 0.2, 1.0],
        beta=0.1,
        num_train_epochs=3,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        tf32=True,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        seed=42,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )


def main(args):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    enable_progress_bars()

    model, processor = load_model_and_processor(MODEL_NAME)
    train_ds, eval_ds = load_dataset(args.dataset_dir, val_split=0.05)
    config = build_dpo_config(args)
    peft_config = None if args.full_finetune else build_lora_config()

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        peft_config=peft_config,
    )
    if peft_config is not None:
        trainer.model.print_trainable_parameters()

    # TODO: --resume_from_checkpoint
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


def parse_args():
    p = argparse.ArgumentParser(description="MPO/DPO fine-tuning for Qwen2.5-VL-3B.")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--output_dir", default="./qwen25vl_oasis_dpo")
    p.add_argument("--run_name", default=None)
    p.add_argument("--full_finetune", action="store_true",
                   help="Full fine-tune instead of LoRA.")
    return p.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    os.makedirs(_args.output_dir, exist_ok=True)
    main(_args)

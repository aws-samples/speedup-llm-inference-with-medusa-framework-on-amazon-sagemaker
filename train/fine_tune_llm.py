# Implementation of this file is based on:
# https://github.com/philschmid/llm-sagemaker-sample/blob/main/scripts/trl/run_sft.py

from dataclasses import dataclass, field
import os
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from peft import LoraConfig

from trl import (
    SFTTrainer,
)


tqdm.pandas()


@dataclass
class ScriptArguments:
    train_dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset, should be /opt/ml/input/data/train_dataset.json"
        },
    )
    eval_dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset, should be /opt/ml/input/data/eval_dataset.json"
        },
    )
    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )
    use_qlora: bool = field(default=False, metadata={"help": "Whether to use QLORA"})
    merge_adapters: bool = field(
        metadata={"help": "Wether to merge weights for LoRA."},
        default=False,
    )
    # Medusa parameters - https://github.com/FasterDecoding/Medusa
    medusa_num_heads: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of heads for the Medusa. If None, Medusa is not used."
        },
    )
    medusa_heads_coefficient: float = field(
        default=0.1, metadata={"help": "Coefficient for Medusa heads"}
    )
    medusa_decay_coefficient: float = field(
        default=1.0, metadata={"help": "Decay coefficient for Medusa"}
    )
    medusa_scheduler: str = field(
        default="constant",
        metadata={
            "help": "Scheduler type for Medusa, currently " "only constant is supported"
        },
    )
    train_only_medusa_heads: bool = field(
        default=True, metadata={"help": "If True, train only medusa heads"}
    )
    medusa_lr_multiplier: float = field(
        default=1.0, metadata={"help": "Learning rate multiplier for Medusa"}
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Dataset
    ################
    train_dataset = load_dataset(
        "json",
        data_files=script_args.train_dataset_path,
        split="train",
    )
    eval_dataset = load_dataset(
        "json",
        data_files=script_args.eval_dataset_path,
        split="train",
    )

    ################
    # Model & Tokenizer
    ################
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    if script_args.use_qlora:
        print("Using QLoRA")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # PEFT
    ################

    # LoRA config
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
    )
    trainer.train()

    ##########################
    # SAVE MODEL FOR SAGEMAKER
    ##########################
    sagemaker_save_dir = "/opt/ml/model"

    trainer.tokenizer.save_pretrained(sagemaker_save_dir)

    if script_args.merge_adapters:
        # merge adapter weights with base model and save
        # save int 4 model
        trainer.model.save_pretrained(training_args.output_dir)
        trainer.tokenizer.save_pretrained(training_args.output_dir)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForCausalLM

        # list file in output_dir
        print(os.listdir(training_args.output_dir))

        # load PEFT model in fp16
        model = AutoPeftModelForCausalLM.from_pretrained(
            training_args.output_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # Merge LoRA and base model and save
        model = model.merge_and_unload()
        model.save_pretrained(
            sagemaker_save_dir, safe_serialization=True, max_shard_size="2GB"
        )
    else:
        trainer.model.save_pretrained(sagemaker_save_dir, safe_serialization=True)

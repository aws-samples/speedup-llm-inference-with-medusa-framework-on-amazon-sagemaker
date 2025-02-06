# Implementation is based on:
# https://github.com/axolotl-ai-cloud/axolotl/compare/main...ctlllll:axolotl:main
# https://github.com/FasterDecoding/Medusa
# https://github.com/philschmid/llm-sagemaker-sample/blob/main/scripts/trl/run_sft.py
import logging
from typing import List, Optional
import os
from pathlib import Path
from contextlib import nullcontext
import shutil

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    Trainer,
)
from transformers.trainer_pt_utils import LabelSmoother
import types
from safetensors.torch import save_file

from trl import SFTTrainer

from dataclasses import dataclass, field

from datasets import load_dataset


IGNORE_TOKEN_ID = LabelSmoother.ignore_index

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.
    Args:
        medusa_num_heads (int): Number of heads for the Medusa layer
        base_model_name_or_path (str): The name or path of the base model.
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=5,
        base_model_name_or_path="/opt/ml/model/base-model/",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = 1
        self.base_model_name_or_path = base_model_name_or_path


class ResBlock(nn.Module):
    """
    A Residual Block module.
    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.
    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


def add_medusa_heads(
    model,
    medusa_num_heads,
):
    """
    Args:
        model (nn.Module): The base language model to be used.
        medusa_num_heads (int, optional): Number of additional tokens to predict
    """
    hidden_size = model.lm_head.weight.shape[-1]
    vocab_size = model.lm_head.weight.shape[0]
    model.config.medusa_num_layers = 1
    model.config.medusa_num_heads = medusa_num_heads
    model.medusa_num_heads = medusa_num_heads
    # Create a list of Medusa heads
    model.medusa_heads = nn.ModuleList(
        [
            nn.Sequential(
                ResBlock(hidden_size),
                nn.Linear(hidden_size, vocab_size, bias=False),
            )
            for _ in range(medusa_num_heads)
        ]
    )

    # Ensure medusa_head's dtype and device align with the base_model
    model.medusa_heads.to(model.dtype).to(model.device)
    logger.info(f"Loading medusa heads in {str(model.dtype)} to device {model.device}")

    for i in range(medusa_num_heads):
        # Initialize the weights of each medusa_head using the base model's weights
        model.medusa_heads[i][-1].weight.data[:] = model.lm_head.weight.data[:]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train_only_medusa_heads: bool = False,
    ):
        """Forward pass of the MedusaModel.
        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        maybe_grad = torch.no_grad() if train_only_medusa_heads else nullcontext()
        with maybe_grad:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            medusa_logits = [self.lm_head(hidden_states)]
        for i in range(self.medusa_num_heads):
            medusa_logits.append(self.medusa_heads[i](hidden_states))
        return torch.stack(medusa_logits, dim=0)

    model.forward = types.MethodType(forward, model)


def freeze_layers(model):
    logger.info("Freeze layers")
    for param in model.parameters():
        param.requires_grad = False

    for param in model.medusa_heads.parameters():
        param.requires_grad = True
    logger.info("Finished freezing layers")


class MedusaSFTTrainer(SFTTrainer):
    def __init__(
        self,
        medusa_num_heads,
        medusa_heads_coefficient,
        medusa_decay_coefficient,
        medusa_scheduler,
        train_only_medusa_heads,
        medusa_lr_multiplier,
        **kwargs,
    ):
        self.medusa_num_heads = medusa_num_heads
        self.medusa_heads_coefficient = medusa_heads_coefficient
        self.medusa_decay_coefficient = medusa_decay_coefficient
        self.medusa_scheduler = medusa_scheduler
        self.train_only_medusa_heads = train_only_medusa_heads
        self.medusa_lr_multiplier = medusa_lr_multiplier

        if getattr(kwargs["model"], "is_quantized", False) and train_only_medusa_heads:
            # Trainer does not know that we will freeze base model layers, and would
            # raise an error that we need an adapter when model is quantized
            # So we trick it to think it is not quantized during init
            setattr(kwargs["model"], "is_quantized", False)
            super().__init__(**kwargs)
            setattr(kwargs["model"], "is_quantized", True)
        else:
            super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        """
        Compute the training loss for the model. Overriding the default with custom loss for Medusa
        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.
        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """

        logits = model(
            **inputs,
            train_only_medusa_heads=self.train_only_medusa_heads,
        )
        labels = inputs["labels"]
        # Shift so that tokens < n predict n
        loss = 0
        loss_fct = CrossEntropyLoss()
        log = {}
        num_heads = logits.shape[0]
        for i in range(num_heads):
            medusa_logits = logits[i, :, : -(1 + i)].contiguous()
            medusa_labels = labels[..., 1 + i:].contiguous()
            medusa_logits = medusa_logits.view(-1, logits.shape[-1])
            medusa_labels = medusa_labels.view(-1)
            medusa_labels = medusa_labels.to(medusa_logits.device)
            loss_i = loss_fct(medusa_logits, medusa_labels)
            # Compute the coefficient for medusa losses
            if self.medusa_scheduler == "constant":
                medusa_scheduler_coefficient = 1
            else:
                raise ValueError(
                    f"Invalid medusa_scheduler: {self.medusa_scheduler}. "
                    "Must be 'constant'."
                )
            if i == 0:
                if not self.train_only_medusa_heads:
                    loss += loss_i
            else:
                loss += (
                    loss_i
                    * self.medusa_decay_coefficient**i
                    * self.medusa_heads_coefficient
                    * medusa_scheduler_coefficient
                )
            not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
            medusa_labels = medusa_labels[not_ignore]

            # Add top-k accuracy
            for k in range(1, 10):
                _, topk = medusa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
                log[f"medusa{i}_top{k}"] = correct.float().mean().item()

            log[f"medusa{i}_loss"] = loss_i.item()
            log["medusa_scheduler_coefficient"] = medusa_scheduler_coefficient
            logger.debug(log)
        return (loss, logits) if return_outputs else loss

    def create_optimizer(self):
        """
        Overriding default method, the only change is in optimizer_grouped_parameters,
        rest is copied from default
        Transformers Trainer default: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/trainer.py#L973
        Medusa optimizer copied from: https://github.com/axolotl-ai-cloud/axolotl/compare/main...ctlllll:axolotl:main#diff-9fc90bdc541619fb6485cd848e4129a7c95197dcb24f352b4ee2027cb9977e44R341
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            # Separately set lr for medusa_head - this is the only change here, other
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n in decay_parameters
                            and p.requires_grad
                            and "medusa_head" not in n
                        )
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n in decay_parameters
                            and p.requires_grad
                            and "medusa_head" in n
                        )
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * self.medusa_lr_multiplier,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2 ** 20}M params")

        return self.optimizer


def save_medusa_heads(final_save_dir, medusa_num_heads, model, torch_dtype):
    """
    Save medusa config, medusa heads, and base model separately
    This is the format that TGI requires for deployment
    """

    # Save medusa config
    medusa_config = MedusaConfig(
        medusa_num_heads=medusa_num_heads,
        base_model_name_or_path="/opt/ml/model/base-model/",
        version="2",
    )
    medusa_config.save_pretrained(final_save_dir)

    # Save medusa heads
    model.medusa_heads.to(torch_dtype)
    logger.info(f"Converting medusa heads to {str(torch_dtype)}")

    state_dict = model.medusa_heads.state_dict()
    save_file(
        state_dict,
        f"{final_save_dir}/medusa_lm_head.safetensors",
    )


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
    model_path: str = field(
        default=None, metadata={"help": "Path to the fine-tuned model"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )
    # Medusa parameters - https://github.com/FasterDecoding/Medusa
    medusa_num_heads: int = field(
        default=5,
        metadata={"help": "Number of heads for the Medusa"},
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
            "help": "Scheduler type for Medusa, currently only constant is supported"
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

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_path,
        device_map="auto",
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
    )

    # Add medusa heads and freeze base model
    add_medusa_heads(
        model,
        medusa_num_heads=script_args.medusa_num_heads,
    )
    freeze_layers(model)
    model.config.torch_dtype = torch_dtype
    model.config.use_cache = False

    logger.info("Finished loading model and medusa heads")

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Training
    ################
    trainer = MedusaSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
        medusa_num_heads=script_args.medusa_num_heads,
        medusa_heads_coefficient=script_args.medusa_heads_coefficient,
        medusa_decay_coefficient=script_args.medusa_decay_coefficient,
        medusa_scheduler=script_args.medusa_scheduler,
        train_only_medusa_heads=script_args.train_only_medusa_heads,
        medusa_lr_multiplier=script_args.medusa_lr_multiplier,
    )
    trainer.train()

    ##########################
    # Save tokenizer, medusa heads, and original model
    ##########################
    sagemaker_save_dir = "/opt/ml/model"

    trainer.tokenizer.save_pretrained(sagemaker_save_dir)
    save_medusa_heads(
        sagemaker_save_dir, script_args.medusa_num_heads, model, torch_dtype
    )

    # Save original base model
    base_model_path = Path("/opt/ml/model/base-model/")
    base_model_path.mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(script_args.model_path):
        file_path = os.path.join(script_args.model_path, filename)
        shutil.move(file_path, base_model_path)

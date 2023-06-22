import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataset import SFTDataset, PretrainDataset
from omegaconf import OmegaConf
from collections import OrderedDict

cfg = OmegaConf.load("./conf.yaml")
model_cfg = cfg.model
dataset_cfg = cfg.dataset

model_id = model_cfg.get("model_id", "baichuan-inc/baichuan-7b")
SAVE_PATH = model_cfg.get("SAVE_PATH", "./outputs/lora/")
max_length = model_cfg.get("max_length", 1024)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token


# Below is the code for loading and preprocessing the model
bnb_config = None
if model_cfg.get("use_qlora", False):
    print("Using QLoRA")
    compute_dtype = {
        "fp16" : torch.float16,
        "bf16" : torch.bfloat16,
        "fp32" : torch.float32
    }
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype[model_cfg.qlora_computetype],
        bnb_4bit_use_double_quant=model_cfg.get("qlora_doublequant", False),
        bnb_4bit_quant_type="nf4",
    )

model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", trust_remote_code=True, quantization_config=bnb_config,
    torch_dtype=torch.float16 if bnb_config is None else bnb_config.bnb_4bit_compute_dtype,)
model.config.use_cache = False


if model_cfg.get("use_lora", False):
    print("Using LoRA")
    from peft import LoraConfig, get_peft_model, TaskType
    if model_cfg.get("use_qlora", False):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=model_cfg.use_grad_checkpoint)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_cfg.get("lora_r", 8),
        lora_alpha=model_cfg.get("lora_alpha", 32),
        lora_dropout=model_cfg.get("lora_dropout", 0.00),
        bias="none",
        target_modules=list(model_cfg.get("lora_module", ["q_proj", "v_proj"])),
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

if model_cfg.get("use_grad_checkpoint", False):
    print("Using gradient checkpointing")
    model.gradient_checkpointing_enable()

# Below is the code for preparing the datasets
train_datasets = OrderedDict()

if dataset_cfg.get("corpus", False):
    print("Using Corpus For Post-Pretrain")
    train_datasets[
        PretrainDataset(
            train_path=dataset_cfg.get("corpus_path", "./datasets/corpus"),
            tokenizer=tokenizer, max_length=max_length
        )] = dataset_cfg.get("corpus_epoch", 3)

if dataset_cfg.get("tough_sft", False):
    print("Using Tough_sft For Instruct-tuning")
    train_datasets[
        SFTDataset(
            train_path=dataset_cfg.get("tough_sft_path", "./datasets/tough_sft/alpaca_data_cleaned.json"),
            tokenizer=tokenizer, max_length=max_length
        )] = dataset_cfg.get("tough_sft_epoch", 1)

if dataset_cfg.get("sft", False):
    print("Using Sft For Instruct-tuning")
    train_datasets[
        SFTDataset(
            train_path=dataset_cfg.get("sft_path", "./datasets/sft/niuyeye.json"),
            tokenizer=tokenizer, max_length=max_length
        )] = dataset_cfg.get("sft_epoch", 3)

# Below is the code for preparing for training process
training_args = transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=-1,
        learning_rate=1e-4,
        adam_beta1=0.9, 
        adam_beta2=0.95,
        fp16=True,
        logging_steps=10,
        output_dir="outputs",
    )

for dataset, num_epochs in train_datasets.items():
    training_args.num_train_epochs = num_epochs

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

model.save_pretrained(SAVE_PATH)

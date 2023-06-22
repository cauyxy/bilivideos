import torch
from peft import PeftModel
from dataset import PROMPT_DICT
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import OmegaConf

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = OmegaConf.load("./conf.yaml")
model_cfg = cfg.model

model_path = model_cfg.get("model_id", "/path/to/raw/model")
peft_model_path = model_cfg.get("SAVE_PATH", "/path/to/peft/model")

tokenizer = AutoTokenizer.from_pretrained(model_path)

input_d = {
    "instruction": "孙悟空最后的结局是什么？",
    "input": ""
}

input_text = PROMPT_DICT["prompt_input"].format_map(input_d) if input_d.get("input", "") != "" else PROMPT_DICT["prompt_no_input"].format_map(input_d)

inputs = tokenizer(input_text, return_tensors="pt").to(device)

base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
output = base_model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0]))

peft_model = PeftModel.from_pretrained(base_model, peft_model_path)
output = peft_model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0]))

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import OmegaConf

cfg = OmegaConf.load("./conf.yaml")
model_cfg = cfg.model

model_path = model_cfg.get("model_id", "/path/to/raw/model")
peft_model_path = model_cfg.get("SAVE_PATH", "/path/to/peft/model")
output_model_path = "./merged_model/" # "/path/of/merged/model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
model_cls = type(base_model)

peft_model = PeftModel.from_pretrained(base_model, peft_model_path)

print(f"Merging with merge_and_unload...")
base_model = peft_model.merge_and_unload()

tokenizer.save_pretrained(output_model_path)
model_cls.save_pretrained(base_model, output_model_path) #, state_dict=deloreanized_sd)

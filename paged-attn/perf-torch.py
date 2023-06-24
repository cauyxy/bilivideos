import torch
from tqdm import tqdm
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("./tmp/torch_trace_" + str(p.step_num) + ".json")

g_config = GenerationConfig(
    top_p=0.95,
    temperature=0.8,
    max_new_tokens=16,
)

model_id = "facebook/opt-125m"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"":0}, torch_dtype=torch.float16)
model.eval()

prompts = [
    "An atom is the basic"
]
prompts = tokenizer(prompts, return_tensors='pt')
prompts = prompts.input_ids.to('cuda')

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
    on_trace_ready=trace_handler
) as p:
    for idx in tqdm(range(8)):
        with torch.no_grad():
            output = model.generate(prompts, generation_config=g_config)

        p.step()

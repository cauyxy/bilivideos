import torch
from torch.profiler import profile, ProfilerActivity
from tqdm import tqdm
from vllm import LLM, SamplingParams

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("./tmp/paged_trace_" + str(p.step_num) + ".json")


prompts = [
    "An atom is the basic"
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=16)

llm = LLM(model="facebook/opt-125m")


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
            outputs = llm.generate(prompts, sampling_params)

        p.step()
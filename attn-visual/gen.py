# Author: Xinyu Yang
# Site: https://yxinyu.com
# Time: 7/1/2023
# Desc: Generate text with attention scores
# python gen.py --model_id facebook/opt-125m --input_text "Once upon a time," --output_file generation_attn.json --remove_special --max_length 10

import json
import torch
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM


global topk
INPUT_Format="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n### Response:\n"

def text_to_idlist(text):
    token_ids = tokenizer(text, return_tensors="pt").input_ids
    token_str = [tid2str(tid) for tid in token_ids[0]]
    return token_ids, token_str


def tid2str(tid):
    raw_str = tokenizer.convert_ids_to_tokens([tid])[0]
    return raw_str


def step(input_ids):
    outputs = model(input_ids, output_attentions=True)
    next_token_logits = outputs.logits[:, -1, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    next_input_ids = torch.cat([input_ids, next_tokens.unsqueeze(0)], dim=-1)
    next_token_str = tid2str(next_tokens[0].item())

    topk_token_ids = torch.topk(next_token_logits, k=topk, dim=-1).indices[0]
    topk_token_strs = [tid2str(tid.item()) for tid in topk_token_ids]
    topk_probs = torch.softmax(next_token_logits, dim=-1)[0][topk_token_ids]
    topk_probs /= topk_probs.sum()
    topk_probs = [round(prob.item(), 4) for prob in topk_probs]

    attn_score_lis = []

    attn_scores = outputs.attentions[-1][0]
    for layer_idx, layer_attn in enumerate(outputs.attentions):
        attn_scores = layer_attn[0]

        layer_score_lis = []
        for idx, head_attn in enumerate(attn_scores):
            head_score = head_attn[-1]
            head_info = {
                "head_name": f"head_{idx}",
                "score_lis": [round(score.item(), 4) for score in head_score],
            }
            layer_score_lis.append(head_info)
        attn_score_lis.append(
            {
                "layer_name": f"layer_{layer_idx}",
                "layer_score_lis": layer_score_lis,
            }
        )

    output_dic = {
        "sentence_idx": len(input_ids[0]),
        "token_id": next_tokens[0].item(),
        "token_str": next_token_str,
        "topk_probs": topk_probs,
        "topk_token_strs": topk_token_strs,
        "attn_score_lis": attn_score_lis,
    }
    return next_input_ids, output_dic


def run_generation(input_text, max_length=10, output_file=None):
    input_ids, prompt_tokens = text_to_idlist(input_text)
    prompt_lis = [
        {
            "sentence_idx": s_idx,
            "token_id": t_id.item(),
            "token_str": t_str,
        }
        for s_idx, (t_id, t_str) in enumerate(zip(input_ids[0], prompt_tokens))
    ]
    output_lis = []
    for _ in range(max_length):
        input_ids, output_dic = step(input_ids)
        if input_ids[0][-1].item() == tokenizer.eos_token_id:
            break
        output_lis.append(output_dic)

    output_file = output_file or f"generation_attn.json"
    with open(output_file, "w", encoding="utf8") as f:
        json.dump(
            {
                "model_name": model.config._name_or_path,
                "head_count": model.config.num_attention_heads,
                "layer_count": model.config.num_hidden_layers,
                "prompt_lis": prompt_lis,
                "output_lis": output_lis,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="facebook/opt-125m")
    parser.add_argument("--input_text", type=str, default="Once upon a time,")
    parser.add_argument("--output_file", type=str, default="generation_attn.json")
    parser.add_argument("--max_length", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    
    topk = args.top_k
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map={"": 0})
    run_generation(INPUT_Format.format(args.input_text), max_length=args.max_length, output_file=args.output_file)

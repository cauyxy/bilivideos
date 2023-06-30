# Author: Xinyu Yang
# Site: https://yxinyu.com
# Time: 7/1/2023
# Desc: Generate text with attention scores
# python gen.py --model_id facebook/opt-125m --input_text "Once upon a time," --output_file generation_attn.json --remove_special --max_length 10

import json
import torch
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM

global remove_special

def text_to_idlist(text):
    token_ids = tokenizer(text, return_tensors="pt").input_ids
    token_str = tokenizer.convert_ids_to_tokens(token_ids[0])
    return token_ids, token_str

def tid2str(tid):
    raw_str = tokenizer.convert_ids_to_tokens([tid])[0]
    if remove_special:
        return raw_str.replace("Ġ", " ").replace("Ċ", "\n")
    return raw_str

def step(input_ids):
    outputs = model(input_ids, output_attentions=True)
    next_token_logits = outputs.logits[:, -1, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    next_input_ids = torch.cat([input_ids, next_tokens.unsqueeze(0)], dim=-1)
    next_token_str = tid2str(next_tokens[0].item())

    attn_score_lis = []

    attn_scores = outputs.attentions[-1][0]
    for layer_idx, layer_attn in enumerate(outputs.attentions):
        attn_scores = layer_attn[0]

        layer_score_lis = []
        for idx, head_attn in enumerate(attn_scores):
            head_score = head_attn[-1]
            head_info = {
                "head_name" : f"head_{idx}",
                "score_lis" : [round(score.item(), 6) for score in head_score],
            }
            layer_score_lis.append(head_info)
        attn_score_lis.append({
            "layer_name" : f"layer_{layer_idx}",
            "layer_score_lis" : layer_score_lis,
        })

    output_dic = {
        "sentence_idx" : len(input_ids[0]),
        "next_token_id" : next_tokens[0].item(),
        "next_token_str" : next_token_str,
        "attn_score_lis" : attn_score_lis,
    }
    return next_input_ids, output_dic

def run_generation(input_text, max_length=10, output_file=None):
    input_ids, prompt_tokens = text_to_idlist(input_text)
    prompt_lis = [
        {
            "sentence_idx" : s_idx,
            "token_id" : t_id.item(),
            "token_str" : t_str,
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
                "model_name" : model.config._name_or_path,
                "head_count" : model.config.num_attention_heads,
                "prompt_lis" : prompt_lis, 
                "output_lis" : output_lis,
            },
            f, indent=4,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="facebook/opt-125m")
    parser.add_argument("--input_text", type=str, default="Once upon a time,")
    parser.add_argument("--output_file", type=str, default="generation_attn.json")
    parser.add_argument("--remove_special", action="store_true")
    parser.add_argument("--max_length", type=int, default=10)
    args = parser.parse_args()

    remove_special = args.remove_special
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map={"":0})
    run_generation(args.input_text, max_length=10, output_file=args.output_file)

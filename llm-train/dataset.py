import io
import json
import torch
from torch.utils.data import Dataset

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


PROMPT_DICT = {
    "prompt_input":
        ("Below is an instruction that describes a task, paired with an input that provides further context. "
         "Write a response that appropriately completes the request.\n\n"
         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"),
    "prompt_no_input": ("Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Response:"),
}


def get_dataset_from_jsonl(jsonl_file, tokenizer=None):
    list_data_dict = jload(jsonl_file)

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    sources = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in list_data_dict
    ]
    targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

    return zip(sources, targets)

class PretrainDataset(Dataset):
    def __init__(self, train_path, tokenizer, max_length=2048):
        self.chunk_list = []
        dataset = jload(train_path)
        self.chunk_list = [
            example["text"]
            for example in dataset
        ]

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.chunk_list)

    def __getitem__(self, idx):
        txt = self.chunk_list[idx]
        encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding='max_length')
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": input_ids,
        }


class SFTDataset(Dataset):
    def __init__(self, train_path, tokenizer, max_length=2048):
        self.sample_list = []
        dataset = get_dataset_from_jsonl(train_path, tokenizer=tokenizer)
        self.sample_list = [s + t for s, t in dataset]

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        txt = self.sample_list[idx]
        encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding='max_length')
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": input_ids,
        }

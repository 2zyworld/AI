import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ELECTRADataset(Dataset):
    def __init__(self, filepath):
        self.dataset = pd.read_csv(filepath, sep="\t", encoding="utf-8")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "monologg/koelectra-small-v3-discriminator"
        )
        print(self.dataset.describe())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        text = row.text
        label = torch.from_numpy(np.asarray(list(row[3:11])))

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length",
            add_special_tokens=True,
        )

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        return input_ids, attention_mask, label

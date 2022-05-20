import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, csv_file):
        self.dataset = pd.read_csv(csv_file, encoding="cp949")
        self.dataset["text"] = self.dataset.apply(
            lambda row: " ".join([x for x in row[2:6] if not pd.isna(x)]), axis=1
        )
        self.dataset = self.dataset.drop(columns="Unnamed: 0")
        self.dataset = self.dataset.drop(['text1', 'text2', 'text3', 'text4'], axis=1)
        self.dataset.iloc[:, 1:9] = self.dataset.iloc[:, 1:9].multiply(0.01)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "monologg/koelectra-small-v3-discriminator"
        )
        print(self.dataset.describe())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        text = row.text
        label = torch.from_numpy(np.asarray(list(row[1:9])))

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


class TextDatasetEncoded(Dataset):
    def __init__(self, csv_file):
        self.dataset = pd.read_csv(csv_file, encoding='utf-8')
        self.tokenizer = AutoTokenizer.from_pretrained(
            "monologg/koelectra-small-v3-discriminator"
        )
        print(self.dataset.describe())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        text = row.text
        label = torch.from_numpy(np.asarray(list(row[:1])))

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

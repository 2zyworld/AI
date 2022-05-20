import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from transformers import (
    ElectraForSequenceClassification,
    ElectraConfig,
    # ElectraModel,
)
from transformers.models.electra.modeling_electra import ElectraClassificationHead


class MyModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.model = ElectraForSequenceClassification.from_pretrained(
            "monologg/koelectra-small-v3-discriminator"
        )

        config = ElectraConfig(hidden_size=256, hidden_dropout_prob=0.1, num_labels=8)
        self.model.classifier = ElectraClassificationHead(config)


class TextLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=n_class, hidden_size=n_hidden, dropout=0.3)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim, 8)
        self.softmax = torch.nn.Softmax()

    def forward(self, x, hidden):
        embs = self.embedding(x)
        out, hidden = self.lstm(embs, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.softmax(out)
        out = out[:, -1]

        return out, hidden

    def init_hidden(self):
        return (torch.zeros(1, batch_size, 32), torch.zeros(1, batch_size, 32))

import torch

from transformers import (
    ElectraForSequenceClassification,
    ElectraConfig,
)
from transformers.models.electra.modeling_electra import ElectraClassificationHead

# kobert part #
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


bertmodel, vocab = get_pytorch_kobert_model()


class KoElectra(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = ElectraForSequenceClassification.from_pretrained(
            "monologg/koelectra-small-v3-discriminator"
        )

        config = ElectraConfig(hidden_size=256, hidden_dropout_prob=0.1, num_labels=8)
        self.model.classifier = ElectraClassificationHead(config)


class TextLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, dropout=0.3)
        self.fc = torch.nn.Linear(hidden_dim, 8)
        self.softmax = torch.nn.Softmax()

    def forward(self, x, hidden):
        embs = self.embedding(x)
        out, hidden = self.lstm(embs, hidden)
        out = self.fc(out)
        out = self.softmax(out)
        out = out[:, -1]

        return out, hidden

    @staticmethod
    def init_hidden(batch_size):
        return torch.zeros(1, batch_size, 32), torch.zeros(1, batch_size, 32)

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

import gluonnlp as nlp
import argparse

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from AI.kobert.data import BERTDataset
from AI.kobert.models import BERTClassifier


# Arguments #
parser = argparse.ArgumentParser(description="Parameters")
parser.add_argument("--max_len", "-ml", type=int, default=64, help="max length of text data")
parser.add_argument("--batch_size", "-b", type=int, default=64, help="batch size")
parser.add_argument("--warmup_ratio", "-wr", type=float, default=0.1, help="warmup ratio")
parser.add_argument("--epochs", "-e", type=int, default=5, help="number of epochs")
parser.add_argument("--max_grad_norm", "-mgn", type=int, help="gradient norm clipping")
parser.add_argument("--log_interval", "-li", type=int, help="interval for training log output")
parser.add_argument("--learning_rate", "-l", type=float, default=5e-5, help="learning rate")
args = parser.parse_args()


# Parameters #
max_len = args.max_len
batch_size = args.batch_size
warmup_ratio = args.warmup_ratio
num_epochs = args.epochs
max_grad_norm = args.max_grad_norm
log_interval = args.log_interval
learning_rate = args.learning_rate


# CUDA #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")


# Model #
bert_model, vocab = get_pytorch_kobert_model()
model = BERTClassifier(bert_model, dr_rate=0.5).to(device)


# Data #
tokenizer = get_tokenizer()
token = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

dataset_train = nlp.data.TSVDataset('train_path', field_indices=[1, 2], num_discard_samples=1)
dataset_valid = nlp.data.TSVDataset('valid_path', field_indices=[1, 2], num_discard_samples=1)

data_train = BERTDataset(dataset_train, 0, 1, token, max_len, True, False)
data_valid = BERTDataset(dataset_valid, 0, 1, token, max_len, True, False)

dataloader_train = DataLoader(data_train, batch_size=batch_size, num_workers=4)
dataloader_valid = DataLoader(data_valid, batch_size=batch_size, num_workers=4)


#
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
criterion = CrossEntropyLoss()

t_total = len(dataloader_train) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


def calculate_accuracy(x, y):
    max_vals, max_indices = torch.max(x, 1)
    train_acc = (max_indices == y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


# Train #
for epoch in range(num_epochs):
    accuracy_train = 0.
    accuracy_valid = 0.
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(dataloader_train):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = criterion(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        accuracy_train += calculate_accuracy(out, label)
        if batch_id % log_interval == 0:
            print(f"{epoch}/{num_epochs} loss {loss.data.cpu().numpy()} accuracy {accuracy_train / (batch_id + 1)}")
    print(f"{epoch}/{num_epochs} | loss {loss.data.cpu().numpy()} | train accuracy {accuracy_train / (batch_id + 1)}")

    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(dataloader_valid):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        accuracy_valid += calculate_accuracy(out, label)
    print(f"{epoch}/{num_epochs} loss {loss.data.cpu().numpy()} | valid accuracy {accuracy_train / (batch_id + 1)}")

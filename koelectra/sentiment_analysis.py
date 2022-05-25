import torch
import argparse

# kobert part #
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from transformers import AdamW

from AI.koelectra.data import TextDataset
from AI.koelectra.models import KoElectra


# Arguments #
parser = argparse.ArgumentParser(description="Hyperparameters")
parser.add_argument("--epochs", "-e", type=int, default=100, help="epochs")
parser.add_argument("--batch_size", "-b", type=int, default=16, help="batch size")
parser.add_argument("--optimizer", "-o", type=str, choices=["CrossEntropy"], default="CrossEntropy", help="optimizer")
parser.add_argument(
    "--learning_rate", "-l", type=float, default=5e-6, help="learning rate"
)
args = parser.parse_args()


# CUDA #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")


# Dataset #
train = TextDataset("training_1.csv")
valid = TextDataset("validation_1.csv")


train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=True)

# Model define #
model = KoElectra().to(device)


# Train #
optimizer = AdamW(model.parameters(), lr=args.learning_rate)
criterion = torch.nn.BCEWithLogitsLoss()

losses = []
for i in range(args.epochs):
    total_loss = 0.
    batches = 0
    model.train()
    for input_ids_batch, attention_masks_batch, y_batch in train_loader:
        optimizer.zero_grad(set_to_none=True)
        y_batch = y_batch.to(device)
        y_hat = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
        y_hat = y_hat.clone().detach().float()

        loss = criterion(y_hat, y_batch)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1
        if batches % 1000 == 0:
            print("Batch loss:", total_loss)

    losses.append(total_loss)
    print("Train loss:", total_loss)


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair
        )
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, idx):
        return (self.sentences[idx] + (self.labels[idx],))

    def __len__(self):
        return (len(self.labels))
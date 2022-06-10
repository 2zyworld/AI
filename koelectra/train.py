import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

import gluonnlp as nlp
import numpy as np
import argparse

from .data import ELECTRADataset
from .models import KoELECTRAModel


# Arguments #
parser = argparse.ArgumentParser(description="Hyperarameters")
parser.add_argument("--batch_size", "-b", type=int, default=16, help="batch size")
parser.add_argument("--epochs", "-e", type=int, default=100, help="epochs")
parser.add_argument(
    "--freeze_layer",
    "-fl",
    type=int,
    default=0,
    choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    help="for hyperparameter tuning choose from what layer to freeze, set 0 to ignore and go full training"
)
parser.add_argument("--learning_rate", "-lr", type=float, default=5e-6, help="learning rate")
parser.add_argument("--cuda", "-c", type=str, default="cpu", choices=["gpu", "cpu"], help="select device for train")
parser.add_argument("--model", "-m", type=str, default=None, help="path to the pretrained model")
args = parser.parse_args()


# Hyperparameters #
max_len = args.max_len
batch_size = args.batch_size
freeze_layer = args.freeze_layer
num_epochs = args.epochs
learning_rate = args.learning_rate
cuda = args.cuda
model_path = args.model


# CUDA #
device = torch.device("cuda:0" if torch.cuda.is_available() and cuda == "gpu" else "cpu")
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")


# Model load #
model = KoELECTRAModel().to(device)
if model_path:
    model.load_state_dict(torch.load(model_path))

trainable = False
for name, param in model.named_parameters():
    if freeze_layer == 0:
        pass
    else:
        if freeze_layer in name:
            trainable = True
        param.requires_grad = trainable


# Dataset #
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

dataset_train = ELECTRADataset("~/dataset/training.tsv")
dataset_valid = ELECTRADataset("~/dataset/validation.tsv")

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=True)


# Optimizer and Loss fn #
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = CrossEntropyLoss()


# Train #
losses = []
accuracies = []
for i in range(args.epochs):
    total_loss = 0.
    correct = 0
    total = 0
    batches = 0

    model.train()
    for input_ids_batch, attention_masks_batch, y_batch in dataloader_train:
        optimizer.zero_grad(set_to_none=True)
        input_ids_batch = input_ids_batch.to(device)
        attention_masks_batch = attention_masks_batch.to(device)
        y_batch = y_batch.to(device)
        y_hat = model(input_ids_batch, attention_mask=attention_masks_batch)[0]
        y_hat = y_hat.clone().detach().float()

        loss = criterion(y_hat, y_batch)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predicted = torch.argmax(y_hat)
        correct += (predicted == y_batch).sum()
        total += len(y_batch)

        batches += 1
        if batches % 500 == 0:
            print("Batch loss:", total_loss)

    losses.append(total_loss)
    accuracies.append(correct.float()/total)
    print("Train loss:", total_loss)

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import argparse

from .data import TextDataset
from .models import MyModel


### Arguments ###
parser = argparse.ArgumentParser(description="Hyperparameters")
parser.add_argument("--epochs", "-e", type=int, default=100, help="epochs")
parser.add_argument("--batch_size", "-b", type=int, default=16, help="batch size")
parser.add_argument(
    "--learning_rate", "-l", type=float, default=5e-6, help="learning rate"
)
args = parser.parse_args()


### CUDA ###
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


### Dataset ###
train_dataset = TextDataset("training.csv")
valid_dataset = TextDataset("validation.csv")
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

### Model define ###
model = MyModel().to(device)


### Train ###
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.MultiLabelSoftMarginLoss()

for i in range(epochs):
    total_loss = 0.
    batches = 0
    model.train()
    for input_ids_batch, attention_masks_batch, y_batch in train_loader:
        optimizer.zero_grad(set_to_none=True)
        y_batch = y_batch.to(device)
        y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
        y_pred = y_pred.clone().detach().float()

        loss = criterion(y_pred, y_batch)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batchs += 1
        if batches % 1000 == 0:
            print("Batch loss:", total_loss)

    losses.append(total_loss)
    print("Train loss:" total_loss)
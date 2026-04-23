import os
from data_loader import get_dataloaders

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(BASE_DIR, "data")

train_loader, val_loader = get_dataloaders(data_dir)

print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))
import torch
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import get_dataloaders
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = get_dataloaders("../data")

model = get_model()
model.load_state_dict(torch.load("../model/pneumonia_model.pth"))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Normal", "Pneumonia"]))
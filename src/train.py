import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_dataloaders
from model import get_model

# ------------------------
# 1. Setup
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "../data"
train_loader, val_loader = get_dataloaders(data_dir)

model = get_model(num_classes=2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ------------------------
# 2. Training config
# ------------------------
num_epochs = 5
best_val_acc = 0.0

# ------------------------
# 3. Training loop
# ------------------------
for epoch in range(num_epochs):

    # -------- TRAIN --------
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # -------- VALIDATION --------
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    # -------- LOGGING --------
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # -------- SAVE BEST MODEL --------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "../model/pneumonia_model.pth")
        print("💾 Best model saved!")

print("\nTraining complete!")
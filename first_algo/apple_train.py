import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"current device: {device}")

train_dir = 'Dataset/processed/Train/Apple_Train'
val_dir = 'Dataset/processed/Val/Apple_Val'
save_dir = os.path.join('first_algo.apple_results', 'Apple_train')
os.makedirs(save_dir, exist_ok=True)

batch_size = 16
epochs = 30
learning_rate = 0.0001
gamma_val = 1.7

# data augmentation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes

# Focal Loss

# setting the alpha; Balancing parameter
class_counts = [len([img for img, label in train_dataset.samples if label == i]) for i in range(len(train_dataset.classes))]

# each balancing parameter is calculated by the reciprocal of number of label images
weights = [1.0 / count for count in class_counts]
# normalization
alpha = torch.tensor(weights) / sum(weights)
alpha = alpha.to(device)

print(f"Classes: {train_dataset.classes}")
print(f"Counts: {class_counts}")
print(f"Alpha weights: {alpha}")

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)  # pt는 맞출 확률
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# EfficientNet V2-S
model = models.efficientnet_v2_s(weights='DEFAULT')

# Fix the classifier; good, imperfect, bad
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 3)
model = model.to(device)

criterion = FocalLoss(alpha=alpha, gamma=1.7)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# for the graph
loss_history = []

# Training
print("Training Started...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    loss_history.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# save the weights
torch.save(model.state_dict(), os.path.join(save_dir, 'apple_weights.pth'))

# Save the Loss record; for the graph
with open(os.path.join(save_dir, 'loss_history.json'), 'w') as f:
    json.dump(loss_history, f)

print(f"Training Complete")

print("\n--- Eval Started ---")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)


        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n[ Classification Report ]")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Apple Confusion Matrix (Gamma={gamma_val})')
plt.savefig(os.path.join(save_dir, 'apple_confusion_matrix.png'))

print(f"\nAll results saved in '{save_dir}' folder.")
plt.show()
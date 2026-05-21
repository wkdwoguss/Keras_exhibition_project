import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"current device: {device}")

train_dir = 'Dataset/processed/Train/Carrot_Train'
val_dir   = 'Dataset/processed/Val/Carrot_Val'
save_dir  = 'first_algo/carrot_results'
os.makedirs(save_dir, exist_ok=True)

batch_size = 64
epochs     = 50
lr         = 0.00001  
gamma_val  = 2.0


class RemappedDataset(Dataset):
    def __init__(self, root, transform, label_map):
        base = datasets.ImageFolder(root)
        self.transform = transform
        self.label_map = label_map

        self.samples = []
        for path, orig_label in base.samples:
            class_name = base.classes[orig_label]
            if class_name in label_map:
                self.samples.append((path, label_map[class_name]))

        self.classes = sorted(label_map, key=lambda k: label_map[k])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def crop_carrot(img):
    img_array = np.array(img)
    mask = img_array.sum(axis=2) > 30
    if mask.sum() < img_array.shape[0] * img_array.shape[1] * 0.05:
        return img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad  = 20
    rmin = max(0, rmin - pad)
    rmax = min(img_array.shape[0], rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(img_array.shape[1], cmax + pad)
    return img.crop((cmin, rmin, cmax, rmax))

def remove_green(img):
    img_array = np.array(img)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    lower_green = np.array([30, 35, 35])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    kernel = np.ones((45, 45), np.uint8)
    dilated_mask = cv2.dilate(green_mask, kernel, iterations=2)
    
    lower_white = np.array([0, 0, 180])   
    upper_white = np.array([180, 60, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    nearby_white = cv2.bitwise_and(dilated_mask, white_mask)
    
    final_mask = cv2.bitwise_or(green_mask, nearby_white)
    img_array[final_mask > 0] = [0, 0, 0]
    
    return Image.fromarray(img_array)


train_transform = transforms.Compose([
    transforms.Lambda(remove_green),
    transforms.Lambda(crop_carrot),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_transform_stage2 = transforms.Compose([
    transforms.Lambda(remove_green),
    transforms.Lambda(crop_carrot),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Lambda(remove_green),
    transforms.Lambda(crop_carrot),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha   = alpha
        self.gamma   = gamma
        self.ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, inputs, targets):
        ce_loss    = self.ce_loss(inputs, targets)
        pt         = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                epochs, label, save_path, patience=10):
    train_loss_history = []
    val_loss_history   = []

    best_val_loss      = float('inf')
    patience_counter   = 0

    print(f"\n{'='*50}")
    print(f"  {label} Train started")
    print(f"{'='*50}")

    for epoch in range(epochs):
        # Train
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(train_loss)

        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_loss_history.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model. (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early Stopping (epoch {epoch+1})")
                break

    return train_loss_history, val_loss_history


def make_loader(dataset, batch_size, shuffle=False, use_sampler=True):
    if use_sampler:
        counts  = {}
        for _, label in dataset.samples:
            counts[label] = counts.get(label, 0) + 1
        weights = [1.0 / counts[label] for _, label in dataset.samples]
        sampler = WeightedRandomSampler(weights, len(weights))
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=8, pin_memory=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=8, pin_memory=True)


print("\n Bad=0  /  Good+Imperfect=1")

stage1_label_map = {'Bad': 0, 'Good': 1, 'Imperfect': 1}

train_ds1 = RemappedDataset(train_dir, train_transform, stage1_label_map)
val_ds1   = RemappedDataset(val_dir,   val_transform,   stage1_label_map)

train_loader1 = make_loader(train_ds1, batch_size, use_sampler=True)
val_loader1   = make_loader(val_ds1,   batch_size, use_sampler=False)

counts1 = [sum(1 for _, l in train_ds1.samples if l == i) for i in range(2)]
print(f"  Bad: {counts1[0]}  /  Not-Bad: {counts1[1]}")

alpha1 = torch.tensor([1.0 / counts1[0], 1.0 / counts1[1]])
alpha1 = (alpha1 / alpha1.sum()).to(device)

model1 = models.efficientnet_v2_s(weights='DEFAULT')
num_ftrs1 = model1.classifier[1].in_features
model1.classifier = nn.Sequential(
    nn.Dropout(p=0.5),              
    nn.Linear(num_ftrs1, 2)
)
if torch.cuda.device_count() > 1:
    model1 = nn.DataParallel(model1)
model1 = model1.to(device)

criterion1 = FocalLoss(alpha=alpha1, gamma=gamma_val)
optimizer1 = optim.Adam(model1.parameters(), lr=lr, weight_decay=1e-3)  
scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=epochs)

train_hist1, val_hist1 = train_model(
    model1, train_loader1, val_loader1, criterion1, optimizer1, scheduler1,
    epochs, "first stage: Bad vs Not-Bad",
    save_path=os.path.join(save_dir, 'stage1_best.pth'),
    patience=10
)

with open(os.path.join(save_dir, 'stage1_loss.json'), 'w') as f:
    json.dump({'train': train_hist1, 'val': val_hist1}, f)


print("\nsecond stage: Good=0  /  Imperfect=1")

stage2_label_map = {'Good': 0, 'Imperfect': 1}

train_ds2 = RemappedDataset(train_dir, train_transform_stage2, stage2_label_map)
val_ds2   = RemappedDataset(val_dir,   val_transform,          stage2_label_map)

train_loader2 = make_loader(train_ds2, batch_size, use_sampler=True)
val_loader2   = make_loader(val_ds2,   batch_size, use_sampler=False)

counts2 = [sum(1 for _, l in train_ds2.samples if l == i) for i in range(2)]
print(f"  Good: {counts2[0]}  /  Imperfect: {counts2[1]}")

alpha2 = torch.tensor([1.0 / counts2[0], 1.0 / counts2[1]])
alpha2 = (alpha2 / alpha2.sum()).to(device)

model2 = models.efficientnet_v2_s(weights='DEFAULT')
num_ftrs2 = model2.classifier[1].in_features
model2.classifier = nn.Sequential(
    nn.Dropout(p=0.5),        
    nn.Linear(num_ftrs2, 2)
)
if torch.cuda.device_count() > 1:
    model2 = nn.DataParallel(model2)
model2 = model2.to(device)

criterion2 = FocalLoss(alpha=alpha2, gamma=gamma_val)
optimizer2 = optim.Adam(model2.parameters(), lr=lr, weight_decay=1e-3) 
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs)

train_hist2, val_hist2 = train_model(
    model2, train_loader2, val_loader2, criterion2, optimizer2, scheduler2,
    epochs, "second stage: Good vs Imperfect",
    save_path=os.path.join(save_dir, 'stage2_best.pth'),
    patience=10
)

with open(os.path.join(save_dir, 'stage2_loss.json'), 'w') as f:
    json.dump({'train': train_hist2, 'val': val_hist2}, f)


print("\nEvaluation started")


model1.load_state_dict(torch.load(os.path.join(save_dir, 'stage1_best.pth')))
model2.load_state_dict(torch.load(os.path.join(save_dir, 'stage2_best.pth')))
model1.eval()
model2.eval()

val_ds_full     = datasets.ImageFolder(val_dir, transform=val_transform)
val_loader_full = DataLoader(val_ds_full, batch_size=batch_size,
                             shuffle=False, num_workers=8, pin_memory=True)

orig_classes = val_ds_full.classes  # ['Bad', 'Good', 'Imperfect']
print(f"Original class order: {orig_classes}")

all_preds  = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader_full:
        inputs = inputs.to(device)

        out1  = model1(inputs)
        pred1 = torch.argmax(out1, dim=1)

        out2  = model2(inputs)
        pred2 = torch.argmax(out2, dim=1)

        final_preds = torch.where(
            pred1 == 0,
            torch.zeros_like(pred1),
            torch.where(
                pred2 == 0,
                torch.ones_like(pred2),
                torch.full_like(pred2, 2)
            )
        )

        all_preds.extend(final_preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n[ Classification Report ]")
print(classification_report(all_labels, all_preds, target_names=orig_classes, digits=4))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=orig_classes, yticklabels=orig_classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Carrot Confusion Matrix (2-Stage)')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'carrot_confusion_matrix_2stage.png'))
plt.show()

print(f"\nAll results saved in '{save_dir}' folder.")
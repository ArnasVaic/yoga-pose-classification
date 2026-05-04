# %%
import kagglehub
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

# %% Download dataset
path = kagglehub.dataset_download("akashrayhan/yoga-82")
print("Path to dataset files:", path)

# %% Number of classes

def count_classes_and_samples(path: Path):
    samples = 0
    for entry in path.iterdir():
        if not entry.is_dir():
            continue
        class_samples = count_folders(entry)
        samples += class_samples
    classes = count_folders(path)
    return samples, classes

def count_folders(path):
    return sum(1 for _ in path.iterdir())

path = Path(path)

for set in ['train', 'test', 'valid']:    
    samples, classes = count_classes_and_samples(path / 'train')
    print(f'{set} size: {samples}, classes: {classes}')
    
# %% Image resolution stats

VALID_EXT = {".jpg", ".jpeg"}

def iter_images(split_path: Path):
    for class_dir in split_path.iterdir():
        if not class_dir.is_dir():
            continue

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in VALID_EXT:
                yield img_path


def get_image_size(img_path: Path):
    with Image.open(img_path) as img:
        return img.size  # (w, h)


def compute_split_stats(split_path: Path):
    widths, heights = [], []

    for img_path in iter_images(split_path):
        try:
            w, h = get_image_size(img_path)
            widths.append(w)
            heights.append(h)
        except Exception:
            continue

    if not widths:
        return None

    widths = np.array(widths)
    heights = np.array(heights)

    return {
        "count": len(widths),
        "min_width": int(widths.min()),
        "max_width": int(widths.max()),
        "avg_width": float(widths.mean()),
        "min_height": int(heights.min()),
        "max_height": int(heights.max()),
        "avg_height": float(heights.mean()),
    }

# ---- main ----

path = Path(path)

for split in ["train", "valid", "test"]:
    stats = compute_split_stats(path / split)

    print(f"\n=== {split.upper()} ===")
    if stats is None:
        print("No images found.")
        continue

    print(f"Images: {stats['count']}")
    print(f"Width : min={stats['min_width']}, max={stats['max_width']}, avg={stats['avg_width']:.1f}")
    print(f"Height: min={stats['min_height']}, max={stats['max_height']}, avg={stats['avg_height']:.1f}")
    
# %%

IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_ds = datasets.ImageFolder(path / "train", transform=transform)
val_ds   = datasets.ImageFolder(path / "valid", transform=transform)
test_ds  = datasets.ImageFolder(path / "test", transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

num_classes = len(train_ds.classes)
print("Classes:", num_classes)

# %%

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.efficientnet_b0(weights="IMAGENET1K_V1")

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(device)

# %%

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# %%

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            preds = out.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return correct / total, all_preds, all_labels

# %%
EPOCHS = 5  # baseline only

for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader)
    acc, _, _ = evaluate(model, val_loader)

    print(f"Epoch {epoch+1}: loss={loss:.4f}, val_acc={acc:.4f}")
    
# %%

acc, preds, labels = evaluate(model, test_loader)

print("Test accuracy:", acc)

print("\nClassification report:")
print(classification_report(labels, preds, target_names=test_ds.classes))

cm = confusion_matrix(labels, preds)
print("\nConfusion matrix shape:", cm.shape)

# %%
report = classification_report(labels, preds, target_names=test_ds.classes, output_dict=True)

weak_classes = sorted(
    [(k, v["f1-score"]) for k, v in report.items() if k in test_ds.classes],
    key=lambda x: x[1]
)

print("Worst classes:")
print(weak_classes[:10])
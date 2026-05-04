# %%
import kagglehub
import os
from pathlib import Path
from PIL import Image
import numpy as np

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

def get_image_size(img_path: Path):
    with Image.open(img_path) as img:
        return img.size  # (width, height)


def get_folder_image_stats(class_path: Path):
    widths = []
    heights = []

    for img_path in class_path.iterdir():
        if not img_path.is_file():
            continue
        try:
            w, h = get_image_size(img_path)
            widths.append(w)
            heights.append(h)
        except Exception:
            # skip corrupted / non-image files
            continue

    if not widths:
        return None

    widths = np.array(widths)
    heights = np.array(heights)

    return {
        "min_width": widths.min(),
        "max_width": widths.max(),
        "avg_width": widths.mean(),
        "min_height": heights.min(),
        "max_height": heights.max(),
        "avg_height": heights.mean(),
        "count": len(widths)
    }


def get_split_stats(split_path: Path):
    stats = {}

    for class_dir in split_path.iterdir():
        if not class_dir.is_dir():
            continue

        class_stats = get_folder_image_stats(class_dir)
        if class_stats:
            stats[class_dir.name] = class_stats

    return stats


# ---- main ----

path = Path(path)

for split in ["train", "test", "valid"]:
    split_path = path / split
    stats = get_split_stats(split_path)

    print(f"\n=== {split.upper()} ===")
    for class_name, s in stats.items():
        print(
            f"{class_name}: "
            f"min={s['min_width']}x{s['min_height']}, "
            f"max={s['max_width']}x{s['max_height']}, "
            f"avg={s['avg_width']:.1f}x{s['avg_height']:.1f}, "
            f"n={s['count']}"
        )
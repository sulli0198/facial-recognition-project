import os
import shutil
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_lfw_people


def prepare_lfw_not_me(
    num_images_to_use=33,         # Total LFW images to fetch
    train_split_ratio=0.8,        # % of images in training set
    lfw_data_home=None            # Download directory (None = default)
):
    """
    Downloads a small subset of the LFW dataset and splits into 'not_me' train/validation folders.

    Args:
        num_images_to_use (int): Total number of LFW images to use.
        train_split_ratio (float): Ratio of images for training (0.8 = 80% train / 20% validation).
        lfw_data_home (str, optional): Custom folder to download LFW dataset. Default: scikit-learn's default.
    """

    # Paths to your project folders
    project_root = Path(__file__).resolve().parent.parent  # Goes up to 'face-recognition-keras'
    train_not_me_dir = project_root / 'data' / 'train' / 'not_me'
    val_not_me_dir = project_root / 'data' / 'validation' / 'not_me'

    # Create target folders if they don't exist
    train_not_me_dir.mkdir(parents=True, exist_ok=True)
    val_not_me_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Downloading LFW dataset...")
    fetch_lfw_people(
        min_faces_per_person=0,
        resize=0.4,                 # Resize images for faster training
        download_if_missing=True,
        data_home=lfw_data_home
    )
    print("LFW dataset downloaded successfully.")

    # Find where LFW images are stored
    if lfw_data_home is None:
        lfw_data_home = Path(os.path.expanduser('~')) / 'scikit_learn_data'
    else:
        lfw_data_home = Path(lfw_data_home)

    lfw_images_dir = lfw_data_home / 'lfw_home' / 'lfw_funneled'

    if not lfw_images_dir.exists():
        raise FileNotFoundError(f"LFW images folder not found at {lfw_images_dir}")

    print("Step 2: Collecting all LFW image paths...")
    all_image_paths = [
        Path(root) / file
        for root, _, files in os.walk(lfw_images_dir)
        for file in files
        if file.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    print(f"Total LFW images found: {len(all_image_paths)}")

    # Limit to the required number of images
    if num_images_to_use > len(all_image_paths):
        print(f"Warning: Requested {num_images_to_use} images, but only {len(all_image_paths)} available.")
        num_images_to_use = len(all_image_paths)

    # Shuffle & pick subset
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(all_image_paths)
    selected_images = all_image_paths[:num_images_to_use]

    # Split into train & validation
    split_index = int(train_split_ratio * len(selected_images))
    train_images = selected_images[:split_index]
    val_images = selected_images[split_index:]

    print(f"Step 3: Copying {len(train_images)} images to training folder...")
    for img_path in train_images:
        shutil.copy(img_path, train_not_me_dir / f"lfw_{img_path.name}")
    print(f"Copied {len(train_images)} images to {train_not_me_dir}")

    print(f"Step 4: Copying {len(val_images)} images to validation folder...")
    for img_path in val_images:
        shutil.copy(img_path, val_not_me_dir / f"lfw_{img_path.name}")
    print(f"Copied {len(val_images)} images to {val_not_me_dir}")

    print("\nLFW preparation complete!")
    print("Next step: Add your 'me' images manually to:")
    print(f"  {project_root / 'data' / 'train' / 'me'}")
    print(f"  {project_root / 'data' / 'validation' / 'me'}")


if __name__ == "__main__":
    prepare_lfw_not_me(
        num_images_to_use=33,       # Number of LFW images
        train_split_ratio=0.8,      # 80% train, 20% validation
        lfw_data_home=None          # Default download location (~/.scikit_learn_data)
    )

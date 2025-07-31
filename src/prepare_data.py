import os
import shutil
from sklearn.datasets import fetch_lfw_people
import numpy as np
from pathlib import Path # Modern way to handle paths

def prepare_lfw_for_not_me_class(
    num_images_to_use=4000,
    train_split_ratio=0.8,
    lfw_data_home=None # Default is ~/.scikit_learn_data/
):
    """
    Downloads LFW, selects a subset, and copies them to 'data/train/not_me'
    and 'data/validation/not_me' folders within the project structure.

    Args:
        num_images_to_use (int): The total number of LFW images to copy
                                 into your 'not_me' folders.
        train_split_ratio (float): The ratio of images to put in 'train' vs 'validation'.
        lfw_data_home (str, optional): Directory to download LFW. Defaults to scikit-learn's default.
                                       Example: '/path/to/my/data_downloads'
    """
    # Define your project's data directories relative to the script's execution location
    # Assumes script is run from the project root (e.g., 'face-recognition/')
    project_root = Path(os.getcwd())
    train_not_me_dir = project_root / 'data' / 'train' / 'not_me'
    val_not_me_dir = project_root / 'data' / 'validation' / 'not_me'

    # Ensure target directories exist, creating them if necessary
    train_not_me_dir.mkdir(parents=True, exist_ok=True)
    val_not_me_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Downloading LFW dataset using scikit-learn...")
    # min_faces_per_person=0 to get all images.
    # resize=0.4 is a common resize value for LFW, adjust if your 'me' photos have different dimensions.
    # download_if_missing=True ensures it's downloaded if not present.
    lfw_people_dataset = fetch_lfw_people(
        min_faces_per_person=0,
        resize=0.4, # Images will be roughly 47*0.4=18px tall, 62*0.4=25px wide if original size is 62x47
        download_if_missing=True,
        data_home=lfw_data_home # Use default or specified download location
    )
    print("LFW download complete.")

    # Determine the actual path where scikit-learn stored the funneled images
    # This is typically in `data_home`/lfw_home/lfw_funneled/
    if lfw_data_home is None:
        # Default path for scikit-learn data
        lfw_data_home = Path(os.path.expanduser('~')) / 'scikit_learn_data'
    else:
        lfw_data_home = Path(lfw_data_home) # Convert string path to Path object

    lfw_source_base_dir = lfw_data_home / 'lfw_home' / 'lfw_funneled'

    if not lfw_source_base_dir.exists():
        print(f"Error: LFW data directory not found at {lfw_source_base_dir}. Please verify scikit-learn's download path or check if download failed.")
        return

    print(f"Step 2: Collecting all LFW image paths from {lfw_source_base_dir}...")
    all_lfw_image_paths = []
    # os.walk is used to efficiently traverse nested directories
    for root, _, files in os.walk(lfw_source_base_dir):
        for file in files:
            # Check for common image file extensions
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                all_lfw_image_paths.append(Path(root) / file) # Append as Path object

    print(f"Found {len(all_lfw_image_paths)} images in the downloaded LFW dataset.")

    # Handle cases where fewer images are found than requested
    if len(all_lfw_image_paths) < num_images_to_use:
        print(f"Warning: Only {len(all_lfw_image_paths)} LFW images are available, but {num_images_to_use} were requested. Using all available images.")
        num_images_to_use = len(all_lfw_image_paths)

    # Shuffle the list of image paths to ensure random selection for subset and split
    np.random.seed(42) # For reproducibility of the shuffle
    np.random.shuffle(all_lfw_image_paths)

    # Select the desired subset of images
    selected_lfw_images = all_lfw_image_paths[:num_images_to_use]

    # Split the selected subset into training and validation sets
    split_index = int(len(selected_lfw_images) * train_split_ratio)
    train_lfw_subset = selected_lfw_images[:split_index]
    val_lfw_subset = selected_lfw_images[split_index:]

    print(f"Step 3: Copying {len(train_lfw_subset)} LFW images to {train_not_me_dir} (training set)...")
    for i, src_path in enumerate(train_lfw_subset):
        # Create a unique destination filename to avoid potential conflicts
        # LFW filenames are usually unique across people already, but this adds robustness.
        dest_filename = f"lfw_{src_path.name}"
        shutil.copy(src_path, train_not_me_dir / dest_filename)
        if (i + 1) % 500 == 0: # Print progress every 500 images
            print(f"  Copied {i + 1} training images...")

    print(f"Step 4: Copying {len(val_lfw_subset)} LFW images to {val_not_me_dir} (validation set)...")
    for i, src_path in enumerate(val_lfw_subset):
        dest_filename = f"lfw_{src_path.name}"
        shutil.copy(src_path, val_not_me_dir / dest_filename)
        if (i + 1) % 100 == 0: # Print progress every 100 images
            print(f"  Copied {i + 1} validation images...")

    print("\n--- LFW image preparation complete! ---")
    print(f"Total LFW images copied: {len(train_lfw_subset) + len(val_lfw_subset)}")
    print(f"Train 'not_me' images copied to: {train_not_me_dir}")
    print(f"Validation 'not_me' images copied to: {val_not_me_dir}")
    print("\nNext, manually add your 'me' images to 'data/train/me' and 'data/validation/me'.")
    print("Ensure you split your 'me' photos into training and validation sets as well!")
    print("Then you can proceed with training your model.")


# --- How to run this script ---
if __name__ == '__main__':
    # --- Configuration Parameters ---
    # Total number of LFW images to use for your 'not_me' class
    # The LFW dataset has over 13,000 images in total.
    # Start with a moderate number like 4000-8000.
    TOTAL_LFW_IMAGES_TO_USE = 4000

    # Ratio of images to put in the training set (e.g., 0.8 means 80% for train, 20% for validation)
    TRAIN_VALIDATION_SPLIT_RATIO = 0.8

    # Optional: Specify a custom directory for LFW download.
    # If None, it defaults to ~/.scikit_learn_data/
    # CUSTOM_LFW_DOWNLOAD_DIR = '/path/to/your/custom_data_dir'
    CUSTOM_LFW_DOWNLOAD_DIR = None

    # Call the main function to start the data preparation
    prepare_lfw_for_not_me_class(
        num_images_to_use=TOTAL_LFW_IMAGES_TO_USE,
        train_split_ratio=TRAIN_VALIDATION_SPLIT_RATIO,
        lfw_data_home=CUSTOM_LFW_DOWNLOAD_DIR
    )
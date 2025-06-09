import os
import shutil
from sklearn.model_selection import train_test_split  # scikit-learn split utility :contentReference[oaicite:1]{index=1}

def split_and_copy(
    source_dir: str,
    output_dir: str,
    test_size: float = 0.1,
    random_state: int = 42
):
    """
    Splits files in source_dir (with subfolders per class) into train/test sets,
    and copies them into output_dir/train/... and output_dir/test/... .
    """
    # 1. Prepare output folders
    train_dir = os.path.join(output_dir, "train")
    test_dir  = os.path.join(output_dir, "test")
    for d in (train_dir, test_dir):
        os.makedirs(d, exist_ok=True)

    # 2. Iterate each class subfolder
    classes = [
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ]  # e.g. ['cat', 'dog', ...] :contentReference[oaicite:2]{index=2}

    for cls in classes:
        cls_src = os.path.join(source_dir, cls)
        # List all files in this class folder
        files = [
            f for f in os.listdir(cls_src)
            if os.path.isfile(os.path.join(cls_src, f))
        ]  # assume images only :contentReference[oaicite:3]{index=3}

        # 3. Split into train/test
        train_files, test_files = train_test_split(
            files,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        # 4. Copy files to new dirs
        for phase, file_list in [("train", train_files), ("test", test_files)]:
            dest_folder = os.path.join(output_dir, phase, cls)
            os.makedirs(dest_folder, exist_ok=True)
            for fname in file_list:
                src_path  = os.path.join(cls_src, fname)
                dst_path  = os.path.join(dest_folder, fname)
                shutil.copy2(src_path, dst_path)

        print(f"Class '{cls}': {len(train_files)} train, {len(test_files)} test")

if __name__ == "__main__":
    SOURCE_DIR = "type"      # e.g. '/content/dataset'
    OUTPUT_DIR = "splitted type"  # e.g. '/content/split_data'
    TEST_SIZE   = 0.1  # 20% for test set
    RANDOM_STATE = 42

    split_and_copy(SOURCE_DIR, OUTPUT_DIR, TEST_SIZE, RANDOM_STATE)

import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(
        brain_transform_hash: str,
        original_data_dir: str,
        output_data_dir: str,
        random_state: int
):

    # Zielstruktur anlegen
    folders = ["Train", "Val", "Test"]
    classes = ["false", "true"] # keep consistent with repo `brain_feature_store`
    for folder in folders:
        for cls in classes:
            os.makedirs(os.path.join(output_data_dir, folder, cls), exist_ok=True)

    # Daten organisieren
    for cls in classes:

        # original files
        original_test_dir = os.path.join(original_data_dir, brain_transform_hash, "test", cls)
        original_trainval_dir = os.path.join(original_data_dir, brain_transform_hash, "trainval", cls)

        test_files = [os.path.join(original_test_dir, f) for f in os.listdir(original_test_dir) if
                      os.path.isfile(os.path.join(original_test_dir, f))]

        trainval_files = [os.path.join(original_trainval_dir, f) for f in os.listdir(original_trainval_dir) if
                          os.path.isfile(os.path.join(original_trainval_dir, f))]
        val_files, train_files = train_test_split(trainval_files, test_size=0.5,
                                                 random_state=random_state)  # 20% Val, 20% Test

        # Dateien kopieren
        for train_file in train_files:
            shutil.copy(train_file, os.path.join(output_data_dir, "Train", cls))
        for val_file in val_files:
            shutil.copy(val_file, os.path.join(output_data_dir, "Val", cls))
        for test_file in test_files:
            shutil.copy(test_file, os.path.join(output_data_dir, "Test", cls))

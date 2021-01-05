import numpy as np
from glob import glob
import random
import shutil
import os
import sys

def createSplittedMergedDataset(datasets_base_dir):
    ''' Returns tuple of directory paths (train_dir, val_dir, test_dir) of the created merged dataset '''

    # settings
    dataset_names = ['Chess-extra-dataset-1', 'Chess-extra-dataset-2', 'Chessman-image-dataset']
    split_names = ['train', 'val', 'test']
    split_ratios = [0.6, 0.3, 0.1]
    random.seed(1234) # for deterministic splitting

    # utility
    def removeDir(path):
        if os.path.exists(path):
            shutil.rmtree(path)
            while os.path.exists(path): # avoiding race condition 
                pass

    def createDir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    # re/create merged dataset directory hierarchy
    merged_dataset_dir = os.path.join(datasets_base_dir, "MergedDataset")
    removeDir(merged_dataset_dir)
    createDir(merged_dataset_dir)
    merged_dataset_dirs = [os.path.join(merged_dataset_dir, split_name) for split_name in split_names]
    for merged_dataset in merged_dataset_dirs:
        createDir(merged_dataset)

    # add all files from all classes from all datasets to MergedDataset
    dataset_dirs = [os.path.join(datasets_base_dir, dataset_name) for dataset_name in dataset_names]
    per_split_counts = [0] * len(split_names)

    for dataset_dir in dataset_dirs:
        class_dirs = glob(os.path.join(dataset_dir, "*"))
        for class_dir in class_dirs:
            class_name = os.path.basename(os.path.normpath(class_dir))
            for split_name in split_names:
                createDir(os.path.join(merged_dataset_dir, split_name, class_name))

            files = glob(os.path.join(class_dir, "*"))
            random.shuffle(files)
            for file in files:
                # choose split directory (train/val/test) to add file to such that the real_split_ratios is closest to split_ratios
                per_split_count = sum(per_split_counts)
                real_split_ratios = [count / max(1, per_split_count) for count in per_split_counts]
                diffs = [split_ratio - real_split_ratio for (split_ratio, real_split_ratio) in zip(split_ratios, real_split_ratios)]
                maxDiffIndex = diffs.index(max(diffs))

                newFileName = str(per_split_counts[maxDiffIndex]) + os.path.splitext(file)[1]
                newFilePath = os.path.join(merged_dataset_dir, split_names[maxDiffIndex], class_name, newFileName)
                shutil.copyfile(file, newFilePath)

                per_split_counts[maxDiffIndex] += 1

    return merged_dataset_dirs


def main():
    datasets_base_dir = sys.argv[1]
    train_dir, val_dir, test_dir = createSplittedMergedDataset(datasets_base_dir)
    print(train_dir, val_dir, test_dir)

if __name__ == "__main__":
    main()


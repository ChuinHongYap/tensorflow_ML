import shutil
from glob import glob
import os
import numpy as np

# move all files to a single directory by getting all files recursively
def move_files(input_path, output_path, file_ext=".jpg"):
  all_files = sorted(glob(input_path+"*/*"+file_ext, recursive = True))
  for files in all_files:
    shutil.move(files, os.path.join(output_path, files.split('/')[-1]))
  
# Iou / Jaccard Index
def iou(pred, true, k = 1):
    intersection = np.sum(pred[true==k])
    sum_matrix = np.add(true,pred)
    sum_matrix[sum_matrix>=k] = k
    union = np.sum(sum_matrix)
    iou = intersection / union
    return iou

# Dice similarity function
def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice


import shutil
from glob import glob

# move all files to a single directory by getting all files recursively
def move_files(input_path, output_path, file_ext=".jpg"):
  all_files = sorted(glob(input_path+"*/*"+file_ext, recursive = True))
  for files in all_files:
    shutil.move(files, OUT_PATH+files.split('/')[-1])
  

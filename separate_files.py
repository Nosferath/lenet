import os
from shutil import copy2
import random

image_dir = "/home/inti/Desktop/Claudio/final_data/"
final_dir = "/home/inti/Desktop/Claudio/Semantic-Segmentation-Suite/crack_images_v2/"
filenames = os.listdir(image_dir + 'original/p/')

indexes = list(range(len(filenames)))
random.seed(42)
random.shuffle(indexes)
filenames = [filenames[i] for i in indexes]

print("Copying train files")
for i in range(0, int(len(filenames)*0.6)):
    copy2(image_dir + 'original/p/' + filenames[i],
          final_dir + "train")
    copy2(image_dir + 'results/p/' + filenames[i],
          final_dir + "train_labels")

print("Copying validation files")
for i in range(int(len(filenames)*0.6), int(len(filenames)*0.9)):
    copy2(image_dir + 'original/p/' + filenames[i],
          final_dir + "val")
    copy2(image_dir + 'results/p/' + filenames[i],
          final_dir + "val_labels")

print("Copying test files")
for i in range(int(len(filenames)*0.9), len(filenames)):
    copy2(image_dir + 'original/p/' + filenames[i],
          final_dir + "test")
    copy2(image_dir + 'results/p/' + filenames[i],
          final_dir + "test_labels")
    
print("Completed")

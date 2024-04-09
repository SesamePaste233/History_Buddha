import os
import shutil

folder = './longxing_dataset'
folder2 = './longxing_dataset_english'

# Iterate over all files in the folder, change substrings based on regex pattern
for filename in os.listdir(folder):
    if filename.endswith('.png'):
        new_filename = filename.replace('头', 'head')\
            .replace('身', 'body')\
            .replace('三尊', 'triplet')\
            .replace('佛', 'buddha')\
            .replace('菩萨', 'pusa')\
            .replace('北魏', 'NorthernWei')\
            .replace('东魏', 'EasternWei')\
            .replace('北齐', 'NorthernQi')\
            .replace('隋', 'Sui')\
            .replace('唐', 'Tang')\
            .replace('晚期', '(Late)')
        #os.rename(os.path.join(folder, filename), os.path.join(folder2, new_filename))
        shutil.copyfile(os.path.join(folder, filename), os.path.join(folder2, new_filename))
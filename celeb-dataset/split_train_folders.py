import os
import random
import shutil

# Supported image extensions
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

# Main race directories
RACE_DIRS = ['caucasian', 'chinese', 'indian', 'malay']

random.seed()

def is_image(filename):
    return any(filename.endswith(ext) for ext in IMAGE_EXTS)

def split_images_in_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if is_image(f) and os.path.isfile(os.path.join(folder_path, f))]
    if not files:
        return
    n_total = len(files)
    n_train = max(1, int(n_total * 0.2))
    train_files = random.sample(files, n_train)
    celeb_name = os.path.basename(folder_path)
    train_folder = os.path.join(os.path.dirname(folder_path), f'{celeb_name}_train')
    os.makedirs(train_folder, exist_ok=True)
    for f in train_files:
        src = os.path.join(folder_path, f)
        dst = os.path.join(train_folder, f)
        shutil.move(src, dst)
    print(f"Moved {len(train_files)} images from {folder_path} to {train_folder}")

def main():
    for race_dir in RACE_DIRS:
        if not os.path.isdir(race_dir):
            continue
        for celeb in os.listdir(race_dir):
            celeb_path = os.path.join(race_dir, celeb)
            if os.path.isdir(celeb_path) and not celeb.endswith('_train'):
                split_images_in_folder(celeb_path)

if __name__ == '__main__':
    main() 
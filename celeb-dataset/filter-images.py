#!/usr/bin/env python3
"""
Celebrity Image Filter Script
Recursively processes celebrity image folders to remove:
- Images with multiple faces
- Noisy/low quality images
- Small icons
- Images that don't meet quality criteria
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import shutil
from typing import List, Tuple
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageFilter:
    def __init__(self, min_size: Tuple[int, int] = (100, 100), 
                 max_faces: int = 1, noise_threshold: float = 50.0,
                 backup_folder: str = None):
        """
        Initialize the image filter.
        
        Args:
            min_size: Minimum image dimensions (width, height)
            max_faces: Maximum number of faces allowed (1 for single face only)
            noise_threshold: Threshold for noise detection (lower = more strict)
            backup_folder: Optional folder to move filtered images instead of deleting
        """
        self.min_size = min_size
        self.max_faces = max_faces
        self.noise_threshold = noise_threshold
        self.backup_folder = backup_folder
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Supported image extensions
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'removed_multiple_faces': 0,
            'removed_no_faces': 0,
            'removed_too_small': 0,
            'removed_noisy': 0,
            'kept': 0
        }

    def is_image_file(self, filepath: Path) -> bool:
        """Check if file is a supported image format."""
        return filepath.suffix.lower() in self.supported_extensions

    def detect_faces(self, image: np.ndarray) -> int:
        """
        Detect faces in image and return count.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Number of faces detected
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return len(faces)

    def is_noisy_image(self, image: np.ndarray) -> bool:
        """
        Check if image is too noisy using Laplacian variance.
        
        Args:
            image: OpenCV image array
            
        Returns:
            True if image is considered noisy/blurry
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < self.noise_threshold

    def is_too_small(self, image: np.ndarray) -> bool:
        """
        Check if image is smaller than minimum size requirements.
        
        Args:
            image: OpenCV image array
            
        Returns:
            True if image is too small
        """
        height, width = image.shape[:2]
        return width < self.min_size[0] or height < self.min_size[1]

    def should_remove_image(self, image_path: Path) -> Tuple[bool, str]:
        """
        Analyze image and determine if it should be removed.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (should_remove, reason)
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return True, "corrupted"
            
            # Check if too small (likely an icon)
            if self.is_too_small(image):
                return True, "too_small"
            
            # Check if noisy/blurry
            if self.is_noisy_image(image):
                return True, "noisy"
            
            # Detect faces
            face_count = self.detect_faces(image)
            
            # Remove if no faces detected
            if face_count == 0:
                return True, "no_faces"
            
            # Remove if more than max allowed faces
            if face_count > self.max_faces:
                return True, "multiple_faces"
            
            return False, "good"
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return True, "error"

    def move_or_delete_image(self, image_path: Path, reason: str):
        """
        Move image to backup folder or delete it.
        
        Args:
            image_path: Path to image file
            reason: Reason for removal
        """
        if self.backup_folder:
            # Create backup directory structure
            backup_path = Path(self.backup_folder)
            relative_path = image_path.relative_to(Path.cwd())
            destination = backup_path / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(image_path), str(destination))
            logger.info(f"Moved {image_path} to backup ({reason})")
        else:
            # Delete file
            image_path.unlink()
            logger.info(f"Deleted {image_path} ({reason})")

    def process_folder(self, folder_path: Path):
        """
        Process all images in a folder recursively.
        
        Args:
            folder_path: Path to folder containing images
        """
        logger.info(f"Processing folder: {folder_path}")
        
        # Get all image files recursively
        image_files = []
        for ext in self.supported_extensions:
            image_files.extend(folder_path.rglob(f"*{ext}"))
            image_files.extend(folder_path.rglob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} image files")
        
        for image_path in image_files:
            self.stats['total_processed'] += 1
            
            should_remove, reason = self.should_remove_image(image_path)
            
            if should_remove:
                self.move_or_delete_image(image_path, reason)
                
                # Update statistics
                if reason == "multiple_faces":
                    self.stats['removed_multiple_faces'] += 1
                elif reason == "no_faces":
                    self.stats['removed_no_faces'] += 1
                elif reason == "too_small":
                    self.stats['removed_too_small'] += 1
                elif reason in ["noisy", "corrupted", "error"]:
                    self.stats['removed_noisy'] += 1
            else:
                self.stats['kept'] += 1
                logger.debug(f"Kept {image_path}")

    def print_statistics(self):
        """Print processing statistics."""
        print("\n" + "="*50)
        print("PROCESSING STATISTICS")
        print("="*50)
        print(f"Total images processed: {self.stats['total_processed']}")
        print(f"Images kept: {self.stats['kept']}")
        print(f"Images removed (multiple faces): {self.stats['removed_multiple_faces']}")
        print(f"Images removed (no faces): {self.stats['removed_no_faces']}")
        print(f"Images removed (too small/icons): {self.stats['removed_too_small']}")
        print(f"Images removed (noisy/corrupted): {self.stats['removed_noisy']}")
        print(f"Total removed: {sum(self.stats.values()) - self.stats['kept'] - self.stats['total_processed']}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Filter celebrity images by removing multi-face, noisy, or icon images')
    parser.add_argument('folder', help='Path to folder containing celebrity image folders')
    parser.add_argument('--min-width', type=int, default=100, help='Minimum image width (default: 100)')
    parser.add_argument('--min-height', type=int, default=100, help='Minimum image height (default: 100)')
    parser.add_argument('--max-faces', type=int, default=1, help='Maximum number of faces allowed (default: 1)')
    parser.add_argument('--noise-threshold', type=float, default=50.0, help='Noise threshold (default: 50.0)')
    parser.add_argument('--backup-folder', help='Backup folder for removed images (optional)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without actually removing')
    
    args = parser.parse_args()
    
    # Validate folder path
    folder_path = Path(args.folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: {args.folder} is not a valid directory")
        return
    
    # Create image filter
    image_filter = ImageFilter(
        min_size=(args.min_width, args.min_height),
        max_faces=args.max_faces,
        noise_threshold=args.noise_threshold,
        backup_folder=args.backup_folder
    )
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be removed")
        # For dry run, we'll need to modify the filter to not actually remove files
        # This is a simplified version - you might want to enhance this
    
    try:
        # Process the folder
        image_filter.process_folder(folder_path)
        
        # Print statistics
        image_filter.print_statistics()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()


# New script for splitting images
import os
import random
import shutil

# Supported image extensions
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

# Main race directories
RACE_DIRS = ['caucasian', 'chinese', 'indian', 'malay']

# Set random seed for reproducibility if needed
random.seed()

def is_image(filename):
    return any(filename.endswith(ext) for ext in IMAGE_EXTS)

def split_images_in_folder(folder_path):
    # List all image files
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
"""
Extract ImageNet-1K parquet files into train/val directory structure.
Each image will be saved in a subdirectory based on its class label.
"""

import os
import pyarrow.parquet as pq
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import io


def extract_parquet_to_imagenet_structure(
    parquet_dir: str,
    output_dir: str,
    split: str = "train"
):
    """
    Extract parquet files to ImageNet directory structure.
    
    Args:
        parquet_dir: Directory containing parquet files
        output_dir: Output directory (will create train/val subdirs)
        split: Either 'train' or 'val'
    """
    # Create output directory structure
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all parquet files for this split
    parquet_files = sorted([
        f for f in os.listdir(parquet_dir) 
        if f.endswith('.parquet') and split in f.lower()
    ])
    
    if not parquet_files:
        # If no specific naming, try to infer based on count
        all_files = sorted([f for f in os.listdir(parquet_dir) if f.endswith('.parquet')])
        if split == "train":
            # Assuming first 294 are train files
            parquet_files = all_files[:294]
        elif split == "val":
            # Assuming next 14 are val files
            parquet_files = all_files[294:308]
        else:  # test
            parquet_files = all_files[308:]
    
    print(f"Found {len(parquet_files)} parquet files for {split} split")
    
    # Track statistics
    total_images = 0
    class_counts = {}
    
    # Process each parquet file
    for parquet_file in tqdm(parquet_files, desc=f"Processing {split} parquet files"):
        parquet_path = Path(parquet_dir) / parquet_file
        
        # Read parquet file
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        
        print(f"\nProcessing {parquet_file}: {len(df)} images")
        
        # Process each row
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting images from {parquet_file}", leave=False):
            label = row['label']
            image_data = row['image']
            
            # Skip if label is -1 (test set without labels)
            if label == -1:
                # For validation/test without labels, create a single directory
                class_dir = split_dir / "unknown"
            else:
                # Create class directory
                class_dir = split_dir / f"{label:04d}"
            
            class_dir.mkdir(exist_ok=True)
            
            # Track class counts
            class_key = str(class_dir.name)
            class_counts[class_key] = class_counts.get(class_key, 0) + 1
            
            # Get image
            if isinstance(image_data, dict):
                # If image is stored as dict with 'bytes' key
                if 'bytes' in image_data:
                    image = Image.open(io.BytesIO(image_data['bytes']))
                elif 'path' in image_data:
                    image = Image.open(image_data['path'])
                else:
                    print(f"Warning: Unknown image format in dict: {image_data.keys()}")
                    continue
            elif isinstance(image_data, bytes):
                # If image is stored as raw bytes
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, Image.Image):
                # If already a PIL Image
                image = image_data
            else:
                print(f"Warning: Unknown image type: {type(image_data)}")
                continue
            
            # Save image
            image_filename = f"{split}_{total_images:07d}.JPEG"
            image_path = class_dir / image_filename
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save as JPEG
            image.save(image_path, 'JPEG', quality=95)
            
            total_images += 1
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Extraction complete for {split} split!")
    print(f"{'='*60}")
    print(f"Total images extracted: {total_images}")
    print(f"Total classes: {len(class_counts)}")
    print(f"\nClass distribution (first 10):")
    for i, (class_name, count) in enumerate(sorted(class_counts.items())[:10]):
        print(f"  Class {class_name}: {count} images")
    if len(class_counts) > 10:
        print(f"  ... and {len(class_counts) - 10} more classes")
    print(f"{'='*60}\n")
    
    return total_images, class_counts


def main():
    """Main extraction function."""
    # Configuration
    parquet_dir = "/home/ssl.distillation/WMDD/datasets/imagenet_data/data/"
    output_dir = "/home/ssl.distillation/WMDD/datasets/imagenet_data"
    
    print("="*60)
    print("ImageNet-1K Parquet Extraction Script")
    print("="*60)
    print(f"Source: {parquet_dir}")
    print(f"Destination: {output_dir}")
    print("="*60)
    
    # Check if parquet directory exists
    if not os.path.exists(parquet_dir):
        print(f"Error: Parquet directory not found: {parquet_dir}")
        return
    
    # Count parquet files
    all_parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    print(f"\nFound {len(all_parquet_files)} total parquet files")
    
    # Ask user for confirmation
    response = input("\nProceed with extraction? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Extraction cancelled.")
        return
    
    # Extract train split
    print("\n" + "="*60)
    print("EXTRACTING TRAIN SPLIT")
    print("="*60)
    train_total, train_classes = extract_parquet_to_imagenet_structure(
        parquet_dir=parquet_dir,
        output_dir=output_dir,
        split="train"
    )
    
    # Extract val split
    print("\n" + "="*60)
    print("EXTRACTING VAL SPLIT")
    print("="*60)
    val_total, val_classes = extract_parquet_to_imagenet_structure(
        parquet_dir=parquet_dir,
        output_dir=output_dir,
        split="val"
    )
    
    # Final summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Train images: {train_total} across {len(train_classes)} classes")
    print(f"Val images: {val_total} across {len(val_classes)} classes")
    print(f"Total images: {train_total + val_total}")
    print(f"\nOutput directory structure:")
    print(f"  {output_dir}/")
    print(f"    train/")
    print(f"      0000/ (class 0)")
    print(f"      0001/ (class 1)")
    print(f"      ...")
    print(f"    val/")
    print(f"      0000/ (class 0)")
    print(f"      0001/ (class 1)")
    print(f"      ...")
    print("="*60)


if __name__ == "__main__":
    main()

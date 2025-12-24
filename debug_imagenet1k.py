"""
Debug script to inspect ImageNet-1K parquet file format.
This helps understand the exact structure of your parquet files.
"""

import pandas as pd
from pathlib import Path
import sys

def inspect_parquet_file(parquet_path):
    """Inspect a single parquet file to understand its structure."""
    print(f"\n{'='*80}")
    print(f"Inspecting: {parquet_path}")
    print(f"{'='*80}")
    
    # Load the parquet file
    df = pd.read_parquet(parquet_path)
    
    print(f"\n1. DataFrame Info:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columns: {list(df.columns)}")
    print(f"   - Number of samples: {len(df)}")
    
    print(f"\n2. Column Types:")
    for col in df.columns:
        print(f"   - {col}: {df[col].dtype}")
    
    print(f"\n3. First Row Details:")
    first_row = df.iloc[0]
    
    for col in df.columns:
        value = first_row[col]
        print(f"\n   Column: {col}")
        print(f"   - Type: {type(value)}")
        print(f"   - Value type: {type(value).__name__}")
        
        if col == 'image':
            print(f"   - Image details:")
            print(f"     * Type: {type(value)}")
            print(f"     * Has 'mode' attr: {hasattr(value, 'mode')}")
            print(f"     * Has 'size' attr: {hasattr(value, 'size')}")
            print(f"     * Has 'convert' method: {hasattr(value, 'convert')}")
            
            # Try to get more info
            if hasattr(value, 'mode'):
                print(f"     * Mode: {value.mode}")
            if hasattr(value, 'size'):
                print(f"     * Size: {value.size}")
            if hasattr(value, '__array__'):
                print(f"     * Has __array__ interface: True")
            
            # If it's a dict, show keys
            if isinstance(value, dict):
                print(f"     * Dictionary keys: {list(value.keys())}")
            
            # Check if it's a PIL Image
            try:
                from PIL import Image
                if isinstance(value, Image.Image):
                    print(f"     * ✓ Is PIL.Image.Image: True")
                    print(f"     * Mode: {value.mode}")
                    print(f"     * Size: {value.size}")
                else:
                    print(f"     * ✗ Is PIL.Image.Image: False")
                    print(f"     * Actual type: {type(value)}")
            except Exception as e:
                print(f"     * Error checking PIL Image: {e}")
        
        elif col == 'label':
            print(f"   - Label value: {value}")
            print(f"   - Label type: {type(value)}")
    
    print(f"\n4. Sample Data (first 3 rows):")
    print(df.head(3))
    
    print(f"\n{'='*80}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_imagenet_parquet.py <path_to_imagenet_data_dir>")
        print("\nExample:")
        print("  python debug_imagenet_parquet.py ~/WMDD/datasets/imagenet_data/data")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1]).expanduser()
    
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)
    
    print(f"\n{'#'*80}")
    print(f"# ImageNet-1K Parquet File Inspector")
    print(f"# Directory: {data_dir}")
    print(f"{'#'*80}")
    
    # Find parquet files
    # Try both structures: with subdirectories or flat
    parquet_files = []
    
    # Check for subdirectory structure
    for split in ['train', 'validation', 'test']:
        split_dir = data_dir / split
        if split_dir.exists():
            files = list(split_dir.glob('*.parquet'))
            if files:
                parquet_files.append((split, files[0]))  # Take first file from each split
                break
    
    # Check for flat structure
    if not parquet_files:
        for pattern in ['train-*.parquet', 'validation-*.parquet', 'test-*.parquet']:
            files = list(data_dir.glob(pattern))
            if files:
                split_name = pattern.split('-')[0]
                parquet_files.append((split_name, files[0]))
                break
    
    if not parquet_files:
        print(f"\nError: No parquet files found in {data_dir}")
        print("\nChecked for:")
        print("  - data_dir/train/*.parquet")
        print("  - data_dir/validation/*.parquet")
        print("  - data_dir/test/*.parquet")
        print("  - data_dir/train-*.parquet")
        print("  - data_dir/validation-*.parquet")
        print("  - data_dir/test-*.parquet")
        sys.exit(1)
    
    # Inspect the first parquet file found
    split_name, parquet_file = parquet_files[0]
    print(f"\nFound parquet files for split: {split_name}")
    inspect_parquet_file(parquet_file)
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print("If the image type is:")
    print("  - PIL.Image.Image → Code should work ✓")
    print("  - dict with 'bytes' → Code handles this ✓")
    print("  - bytes → Code handles this ✓")
    print("  - numpy array → Code handles this ✓")
    print("  - Other object → May need custom handling")
    print("\nIf you see errors, share the output above for debugging.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

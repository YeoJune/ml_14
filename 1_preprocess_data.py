"""
Step 1: Data Preprocessing
Download and process Pluribus dataset
"""
import sys
import os
sys.path.append('src')

import numpy as np
from dataset import (
    download_pluribus_files,
    process_phh_files,
    save_processed_data
)

# Configuration
CONFIG = {
    'n_files': None,  # None = use all files, or specify number like 100 for testing
    'use_cache': True,
    'cache_path': 'data/raw',
    'output_path': 'data/processed',
    'target_player': 'p1'
}

def main():
    print("="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    # Download PHH files
    print("\n1. Downloading Pluribus dataset...")
    phh_texts = download_pluribus_files(
        n_files=CONFIG['n_files'],
        use_cache=CONFIG['use_cache'],
        cache_path=CONFIG['cache_path']
    )
    
    print(f"\n✓ Downloaded {len(phh_texts)} PHH files")
    
    # Process into features and labels
    print("\n2. Processing decision points...")
    features, labels = process_phh_files(phh_texts, target_player=CONFIG['target_player'])
    
    print(f"\n✓ Extracted {len(features)} decision points")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    # Show class distribution
    print("\n3. Class distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    action_names = ['fold', 'check_call', 'raise_small', 'raise_medium', 'raise_large', 'all_in']
    for label, count in zip(unique, counts):
        pct = count / len(labels) * 100
        print(f"  {action_names[label]:<15}: {count:>7} ({pct:>5.2f}%)")
    
    # Save processed data
    print("\n4. Saving processed data...")
    save_processed_data(features, labels, output_path=CONFIG['output_path'])
    
    print("\n" + "="*60)
    print("✓ PREPROCESSING COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()

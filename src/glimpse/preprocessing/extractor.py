"""Extract features from raw LC-MS data (mzML/mzXML).

This module uses pyteomics to read raw mass spectrometry files and bins the m/z
axis into fixed windows, taking the maximum intensity within each window.
"""

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from pyteomics import mzml, mzxml

def bin_spectrum(mz_array: np.ndarray, intensity_array: np.ndarray, mz_min: float = 50.0, mz_max: float = 1000.0, bin_size: float = 1.0) -> np.ndarray:
    """Bin a single spectrum into fixed m/z windows and return max intensity per bin."""
    n_bins = int((mz_max - mz_min) / bin_size)
    bins = np.zeros(n_bins)
    
    # ایجاد اندیس‌های بازه‌ها
    bin_indices = np.floor((mz_array - mz_min) / bin_size).astype(int)
    
    # فیلتر کردن پیک‌هایی که در محدوده ما هستند
    valid_mask = (bin_indices >= 0) & (bin_indices < n_bins)
    valid_bins = bin_indices[valid_mask]
    valid_intensities = intensity_array[valid_mask]
    
    # گرفتن بیشترین شدت در هر بازه
    for idx, intensity in zip(valid_bins, valid_intensities):
        if intensity > bins[idx]:
            bins[idx] = intensity
            
    return bins

def process_mzml_file(file_path: Path, mz_min: float = 50.0, mz_max: float = 1000.0, bin_size: float = 1.0) -> np.ndarray:
    """Process a single mzML file and return a binned feature vector."""
    total_bins = np.zeros(int((mz_max - mz_min) / bin_size))
    
    try:
        # خواندن فایل mzML
        with mzml.MzML(str(file_path)) as reader:
            for spectrum in reader:
                if 'm/z array' in spectrum and 'intensity array' in spectrum:
                    mz_array = spectrum['m/z array']
                    intensity_array = spectrum['intensity array']
                    
                    binned = bin_spectrum(mz_array, intensity_array, mz_min, mz_max, bin_size)
                    total_bins = np.maximum(total_bins, binned) # ترکیب طیف‌ها
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        
    return total_bins

def build_dataset_from_directory(directory_path: Path, labels: dict, mz_min: float = 50.0, mz_max: float = 1000.0, bin_size: float = 1.0) -> pd.DataFrame:
    """Process all mzML files in a directory and build a DataFrame with labels.
    
    Args:
        directory_path: Path to the folder containing .mzML files.
        labels: A dictionary mapping filenames to their label (0 for control, 1 for disease).
    """
    data = []
    feature_names = [f"mz_{int(mz_min + i*bin_size)}" for i in range(int((mz_max - mz_min) / bin_size))]
    
    for file_path in directory_path.glob("*.mzML"):
        print(f"Processing {file_path.name}...")
        features = process_mzml_file(file_path, mz_min, mz_max, bin_size)
        
        # اگر فایل در لیبل‌ها نبود، آن را نادیده بگیر
        if file_path.name not in labels:
            print(f"Warning: No label found for {file_path.name}. Skipping.")
            continue
            
        row = {"filename": file_path.name, "label": labels[file_path.name]}
        row.update(dict(zip(feature_names, features)))
        data.append(row)
        
    df = pd.DataFrame(data)
    return df
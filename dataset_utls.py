import imports
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# dataset_utils specific imports
from typing import Optional, List
from joblib import Parallel, delayed
from tqdm import tqdm 
from collections import Counter

def find_edf_files(root_dir: str, montage: str = '_tcp_ar', epilepsy: bool = True) -> tuple:
    path2edf = []
    edf_files_dict = {}
    empty_dirs = []
    if epilepsy:
        identifier = '00_epilepsy'
    else:
        identifier = '01_no_epilepsy'
    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if the directory is empty
        if not dirnames and not filenames:
            #print(f"Directory {dirpath} is empty.")
            continue
        # Check if the directory contains EDF files
        if not any(file.lower().endswith('.edf') for file in filenames):
            #print(f"No EDF files found in {dirpath}.")
            empty_dirs.append(dirpath)
            continue
    
        for file in filenames:
            if file.lower().endswith('.edf'):
                if dirpath.find(montage) != -1 and dirpath.find(identifier) != -1:
                    path2edf.append(os.path.join(dirpath, file)) 
                    edf_files_dict[file] = os.path.join(dirpath, file)

    return path2edf, edf_files_dict, empty_dirs

def read_edf_metadata(edf_path: str) -> dict:
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose='ERROR')
        info = raw.info
        metadata = {
            'file_path': edf_path,
            'n_channels': info['nchan'],
            'sample_rate': info['sfreq'],
            'duration_sec': raw.n_times / info['sfreq'],
            'n_samples': raw.n_times,
            'channel_names': info['ch_names'],
            'channel_positions': info['chs']
            #'events': raw.annotations.events,
        }
        return metadata
    except Exception as e:
        print(f"Failed to read {edf_path}: {e}")
        return None
    
def get_metadata_from_files(edf_files: list, save2csv: bool = False, output_file: str = './info_files/metadata.txt') -> list:
    metadata_list = []
    for edf_file in edf_files:
        metadata = read_edf_metadata(edf_file)
        if metadata:
            metadata_list.append(metadata)
    if save2csv:
        df = pd.DataFrame(metadata_list)
        df.to_csv(output_file, index=False)
        print(f"Metadata saved to {output_file}")
    return metadata_list

def compute_metadata_statistics(metadata_list: list, rounding:int = 2) -> dict:
    stats = {}
    # 1. Statistics for 'duration_sec'
    durations = [m['duration_sec'] for m in metadata_list]
    stats['duration_sec'] = {
        'mean': np.mean(durations).round(rounding) if durations else None,
        'std': np.std(durations).round(rounding) if durations else None,
        'max': np.max(durations) if durations else None,
        'min': np.min(durations) if durations else None
    }
    
    # 2. Statistics for 'n_channels'
    n_channels = [m['n_channels'] for m in metadata_list]
    stats['n_channels'] = {
        'mean': np.mean(n_channels).round(rounding) if n_channels else None,
        'max': np.max(n_channels) if n_channels else None,
        'min': np.min(n_channels) if n_channels else None
    }
    
    # 3. Statistics for 'n_samples'
    n_samples = [m['n_samples'] for m in metadata_list]
    stats['n_samples'] = {
        'mean': np.mean(n_samples).round(rounding) if n_samples else None,
        'std': np.std(n_samples).round(rounding) if n_samples else None,
        'max': np.max(n_samples) if n_samples else None,
        'min': np.min(n_samples) if n_samples else None
    }
    
    # 4. Unique 'sample_rate' values
    sample_rates = {m['sample_rate'] for m in metadata_list}
    stats['sample_rate'] = sorted(sample_rates)  # Sorted for readability
    
    # 5. Shared channels across all files
    if not metadata_list:
        stats['shared_channels'] = []
    else:
        common_channels = set(metadata_list[0]['channel_names'])
        for m in metadata_list[1:]:
            common_channels.intersection_update(m['channel_names'])
            if not common_channels:
                break  # Early exit if no common channels
        stats['shared_channels'] = sorted(common_channels)  # Sorted for readability
    
    return stats

def create_stats_dataframe(stats_dict: dict) -> pd.DataFrame:
    # Initialize a list to hold rows for the DataFrame
    rows = []
    
    # Process numerical metrics with mean/std/min/max
    numerical_metrics = ['duration_sec', 'n_channels', 'n_samples']
    for metric in numerical_metrics:
        row = {
            'metric': metric,
            'mean': stats_dict[metric].get('mean'),
            'std': stats_dict[metric].get('std'),
            'max': stats_dict[metric].get('max'),
            'min': stats_dict[metric].get('min'),
            'unique_values': None,
            'shared_channels': None
        }
        rows.append(row)
    
    # Process sample_rate (unique values)
    rows.append({
        'metric': 'sample_rate',
        'mean': None,
        'std': None,
        'max': None,
        'min': None,
        'unique_values': stats_dict['sample_rate'],
        'shared_channels': None
    })
    
    # Process shared_channels (common across files)
    rows.append({
        'metric': 'shared_channels',
        'mean': None,
        'std': None,
        'max': None,
        'min': None,
        'unique_values': None,
        'shared_channels': stats_dict['shared_channels']
    })
    
    # Create DataFrame
    df_stats = pd.DataFrame(rows)
    return df_stats

def compute_channel_stats(metadata_list):
    # Flatten all channel names across files
    all_channels = [channel for m in metadata_list for channel in m['channel_names']]
    channel_freq = Counter(all_channels)
    
    # Identify shared channels (common across all files)
    if not metadata_list:
        shared_channels = set()
    else:
        shared_channels = set(metadata_list[0]['channel_names'])
        for m in metadata_list[1:]:
            shared_channels.intersection_update(m['channel_names'])
    
    # Create DataFrame with frequency and shared status
    df_channels = pd.DataFrame({
        'channel': list(channel_freq.keys()),
        'frequency': list(channel_freq.values()),
        'is_shared': [ch in shared_channels for ch in channel_freq.keys()]
    })
    
    # Sort by frequency (descending)
    df_channels = df_channels.sort_values(by='frequency', ascending=False)
    
    return df_channels

def plot_channel_frequencies(df_channels):
    plt.figure(figsize=(10, 12))
    palette = {True: 'green', False: 'gray'}  # Highlight shared channels in green
    
    sns.barplot(
        x='frequency',
        y='channel',
        data=df_channels,
        hue='is_shared',
        palette=palette,
        dodge=False  # Avoid splitting bars for True/False
    )
    
    plt.title('Channel Frequencies (Shared Channels Highlighted)')
    plt.xlabel('Number of Files Containing Channel')
    plt.ylabel('Channel Name')
    plt.legend(title='Shared Across All Files', labels=['No', 'Yes'])
    plt.show()


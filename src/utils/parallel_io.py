"""
Parallel I/O utilities for efficient ROOT file handling.

This module provides functions for opening multiple ROOT files in parallel
to significantly speed up I/O operations.
"""

import uproot
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Callable
import time
import os
from pathlib import Path

from .parallel_config import get_optimal_worker_count, DEFAULT_IO_WORKERS


def open_single_file_with_error_handling(args: Tuple[str, str, str]) -> Tuple[Optional[object], str, Optional[str]]:
    """
    Open a single ROOT file with error handling.
    
    Parameters:
    -----------
    args : tuple
        (base_path, filename_pattern, seed) tuple
        
    Returns:
    --------
    tuple: (tree_object, seed, error_message)
        tree_object is None if file failed to open
        error_message is None if successful
    """
    base_path, filename_pattern, seed = args
    
    try:
        filepath = base_path + filename_pattern.format(seed)
        
        # Check if file exists first
        if not os.path.exists(filepath):
            return None, seed, f"File does not exist: {filepath}"
        
        # Open the file and get the events tree
        tree = uproot.open(filepath + ':events')
        return tree, seed, None
        
    except Exception as e:
        return None, seed, f"Failed to open file for seed {seed}: {str(e)}"


def open_files_parallel(base_path: str, filename_pattern: str, seeds: List, 
                       max_workers: Optional[int] = None, 
                       progress_callback: Optional[Callable] = None) -> Tuple[List[object], List[str]]:
    """
    Open multiple ROOT files in parallel.
    
    Parameters:
    -----------
    base_path : str
        Base directory path for files
    filename_pattern : str
        Filename pattern with {} placeholder for seed
    seeds : list
        List of seed values
    max_workers : int, optional
        Maximum number of worker threads (default: min(32, len(seeds), cpu_count+4))
    progress_callback : callable, optional
        Function to call with progress updates (current_count, total_count)
        
    Returns:
    --------
    tuple: (successful_trees, failed_seeds)
        successful_trees: list of opened tree objects in same order as seeds
        failed_seeds: list of seeds that failed to open
    """
    
    if max_workers is None:
        # Use optimized worker count
        max_workers = get_optimal_worker_count(len(seeds), io_bound=True)
    
    print(f"Opening {len(seeds)} files using {max_workers} parallel workers...")
    
    # Prepare arguments for parallel processing
    args_list = [(base_path, filename_pattern, seed) for seed in seeds]
    
    # Track results - we need to maintain order
    results = [None] * len(seeds)
    failed_seeds = []
    completed_count = 0
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and map them to their original indices
        future_to_index = {
            executor.submit(open_single_file_with_error_handling, args): i 
            for i, args in enumerate(args_list)
        }
        
        # Process completed tasks
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            tree, seed, error = future.result()
            
            completed_count += 1
            
            if tree is not None:
                results[index] = tree
            else:
                failed_seeds.append(seed)
                print(f"Warning: {error}")
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(completed_count, len(seeds))
            
            # Print progress every 50 files or at the end
            if completed_count % 50 == 0 or completed_count == len(seeds):
                elapsed = time.time() - start_time
                rate = completed_count / elapsed if elapsed > 0 else 0
                print(f"Progress: {completed_count}/{len(seeds)} files opened "
                      f"({rate:.1f} files/sec, {elapsed:.1f}s elapsed)")
    
    # Filter out None results (failed opens)
    successful_trees = [tree for tree in results if tree is not None]
    
    total_time = time.time() - start_time
    success_rate = len(successful_trees) / len(seeds) * 100
    
    print(f"Parallel file opening completed in {total_time:.2f}s")
    print(f"Successfully opened: {len(successful_trees)}/{len(seeds)} files ({success_rate:.1f}%)")
    
    if failed_seeds:
        print(f"Failed to open {len(failed_seeds)} files: {failed_seeds[:10]}{'...' if len(failed_seeds) > 10 else ''}")
    
    return successful_trees, failed_seeds


def open_files_in_chunks(base_path: str, filename_pattern: str, seeds: List,
                        chunk_size: int = 100, max_workers: Optional[int] = None,
                        progress_callback: Optional[Callable] = None) -> Tuple[List[List[object]], List[str]]:
    """
    Open files in chunks to manage memory usage.
    
    Parameters:
    -----------
    base_path : str
        Base directory path for files
    filename_pattern : str  
        Filename pattern with {} placeholder for seed
    seeds : list
        List of seed values
    chunk_size : int
        Number of files to open in each chunk
    max_workers : int, optional
        Maximum number of worker threads per chunk
    progress_callback : callable, optional
        Function to call with progress updates
        
    Returns:
    --------
    tuple: (chunks_of_trees, all_failed_seeds)
        chunks_of_trees: list of lists, each containing opened trees for a chunk
        all_failed_seeds: combined list of all seeds that failed to open
    """
    
    print(f"Opening {len(seeds)} files in chunks of {chunk_size}...")
    
    chunks_of_trees = []
    all_failed_seeds = []
    total_processed = 0
    
    # Split seeds into chunks
    for i in range(0, len(seeds), chunk_size):
        chunk_seeds = seeds[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        total_chunks = (len(seeds) + chunk_size - 1) // chunk_size
        
        print(f"\nProcessing chunk {chunk_num}/{total_chunks} ({len(chunk_seeds)} files)...")
        
        # Open files in this chunk
        chunk_trees, chunk_failed = open_files_parallel(
            base_path, filename_pattern, chunk_seeds, 
            max_workers=max_workers,
            progress_callback=None  # Handle progress at chunk level
        )
        
        chunks_of_trees.append(chunk_trees)
        all_failed_seeds.extend(chunk_failed)
        
        total_processed += len(chunk_seeds)
        
        if progress_callback:
            progress_callback(total_processed, len(seeds))
    
    return chunks_of_trees, all_failed_seeds


def validate_file_paths(base_path: str, filename_pattern: str, seeds: List) -> Tuple[List, List]:
    """
    Validate that files exist before attempting to open them.
    
    Parameters:
    -----------
    base_path : str
        Base directory path
    filename_pattern : str
        Filename pattern with {} placeholder  
    seeds : list
        List of seed values
        
    Returns:
    --------
    tuple: (existing_seeds, missing_seeds)
    """
    existing_seeds = []
    missing_seeds = []
    
    print(f"Validating {len(seeds)} file paths...")
    
    for seed in seeds:
        filepath = base_path + filename_pattern.format(seed)
        if os.path.exists(filepath):
            existing_seeds.append(seed)
        else:
            missing_seeds.append(seed)
    
    if missing_seeds:
        print(f"Warning: {len(missing_seeds)} files do not exist")
        if len(missing_seeds) <= 10:
            print(f"Missing files for seeds: {missing_seeds}")
        else:
            print(f"Missing files for seeds: {missing_seeds[:10]}... (and {len(missing_seeds)-10} more)")
    
    print(f"Found {len(existing_seeds)} existing files")
    return existing_seeds, missing_seeds


def estimate_memory_usage(base_path: str, filename_pattern: str, sample_seeds: List, 
                         num_samples: int = 5) -> dict:
    """
    Estimate memory usage by sampling a few files.
    
    Parameters:
    -----------
    base_path : str
        Base directory path
    filename_pattern : str
        Filename pattern
    sample_seeds : list
        List of seeds to sample from
    num_samples : int
        Number of files to sample
        
    Returns:
    --------
    dict: Memory usage statistics
    """
    import psutil
    import gc
    
    if len(sample_seeds) < num_samples:
        num_samples = len(sample_seeds)
    
    sample_seeds = sample_seeds[:num_samples]
    
    print(f"Estimating memory usage from {num_samples} sample files...")
    
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    trees = []
    for seed in sample_seeds:
        try:
            filepath = base_path + filename_pattern.format(seed)
            tree = uproot.open(filepath + ':events')
            trees.append(tree)
        except Exception as e:
            print(f"Warning: Could not open sample file for seed {seed}: {e}")
    
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    memory_per_file = (final_memory - initial_memory) / len(trees) if trees else 0
    
    # Clean up
    for tree in trees:
        try:
            tree.file.close()
        except Exception:
            pass
    del trees
    gc.collect()
    
    stats = {
        'memory_per_file_mb': memory_per_file,
        'estimated_total_mb': memory_per_file * len(sample_seeds),
        'current_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
        'available_memory_mb': psutil.virtual_memory().available / 1024 / 1024
    }
    
    print(f"Estimated memory per file: {memory_per_file:.1f} MB")
    print(f"Estimated total memory for all files: {stats['estimated_total_mb']:.1f} MB")
    print(f"Available system memory: {stats['available_memory_mb']:.1f} MB")
    
    return stats 
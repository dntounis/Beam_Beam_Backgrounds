"""
Configuration for parallel processing parameters.
"""

import os

def get_optimal_worker_count(num_files: int, io_bound: bool = True) -> int:
    """Get optimal number of workers based on system resources."""
    cpu_count = os.cpu_count()
    
    if io_bound:
        # For I/O bound operations, use more workers than CPU cores
        optimal = min(32, cpu_count + 4, num_files)
    else:
        # For CPU bound operations, use number of CPU cores
        optimal = min(cpu_count, num_files)
    
    return max(1, optimal)

# Default configuration
DEFAULT_IO_WORKERS = get_optimal_worker_count(100, io_bound=True)
DEFAULT_CHUNK_SIZE = 100
 
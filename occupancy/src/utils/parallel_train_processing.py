"""
Parallel train processing utilities for DD4hep hit analysis.

This module provides functions for processing multiple trains in parallel
using multiprocessing to significantly speed up train-based analysis.
"""

import multiprocessing as mp
from multiprocessing import Pool, Manager
from functools import partial
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple
import pickle

from src.utils.parallel_io import open_files_parallel
from src.utils.parallel_config import get_optimal_worker_count


class _SingleEventView:
    """Light wrapper to expose a single-entry view of an uproot.TTree.

    Provides an arrays(...) method compatible with uproot while always
    selecting a specific entry range [idx, idx+1).
    """
    def __init__(self, tree, entry_idx: int):
        self._tree = tree
        self._idx = int(entry_idx)

    def arrays(self, names, *args, **kwargs):
        # Force single-entry slice for this view
        kwargs = dict(kwargs)
        kwargs.setdefault('entry_start', self._idx)
        kwargs.setdefault('entry_stop', self._idx + 1)
        # Ensure awkward output by default
        kwargs.setdefault('library', 'ak')
        return self._tree.arrays(names, *args, **kwargs)


def process_single_train_worker(args: Tuple) -> Tuple[int, Optional[Dict], Optional[str]]:
    """
    Worker function to process a single train.
    
    This function is designed to be called by multiprocessing workers.
    
    Parameters:
    -----------
    args : tuple
        (train_idx, train_seeds, detector_name, detector_config, 
         buffer_depths, xml_file, constants, main_xml, 
         base_path, filename_pattern, remove_zeros, time_cut, 
         calo_hit_time_def, energy_thresholds, hpp_file, hpp_mu,
         use_vectorized)
    
    Returns:
    --------
    tuple: (train_idx, train_stats, error_message)
        train_stats is None if processing failed
        error_message is None if successful
    """
    try:
        (train_idx, train_seeds, detector_name, detector_config, 
         buffer_depths, xml_file, constants, main_xml, 
         base_path, filename_pattern, remove_zeros, time_cut, 
         calo_hit_time_def, energy_thresholds, hpp_file, hpp_mu,
         use_vectorized) = args
        
        print(f"Worker processing train {train_idx+1} with {len(train_seeds)} files...")

        train_trees = []
        try:
            # Open files for this train using parallel I/O
            train_trees, failed_seeds = open_files_parallel(
                base_path, filename_pattern, train_seeds,
                max_workers=min(8, len(train_seeds))  # Limit workers per train to avoid resource conflicts
            )

            if failed_seeds:
                print(f"Warning: Train {train_idx+1} failed to open {len(failed_seeds)} files")

            if not train_trees:
                return train_idx, None, f"No files successfully opened for train {train_idx+1}"

            # Import here to avoid circular imports and pickling issues
            from src.hit_analysis.occupancy import analyze_detector_hits_vectorized

            # Analyze this train using the requested processing mode
            # IPC-only stats
            ipc_stats = analyze_detector_hits_vectorized(
                train_trees, detector_name, detector_config,
                buffer_depths, xml_file, constants, main_xml,
                remove_zeros, time_cut, calo_hit_time_def, energy_thresholds,
                use_vectorized=use_vectorized
            )
            try:
                ipc_time_len = len(ipc_stats.get('times', [])) if ipc_stats else -1
                print("JIM DEBUG [train-worker]", detector_name, "train", train_idx+1,
                      "times length:", ipc_time_len)
            except Exception:
                pass

            # Optionally build HPP event views and stats
            hpp_stats = None
            combined_stats = None
            if hpp_file is not None and isinstance(hpp_mu, (int, float)) and hpp_mu >= 0:
                import uproot
                import numpy as np
                try:
                    with uproot.open(hpp_file) as hf:
                        htree = hf["events"]
                        num_hpp_events = int(getattr(htree, 'num_entries', 0))
                        if num_hpp_events <= 0:
                            print(f"Warning: HPP file {hpp_file} has no events; skipping HPP overlay for train {train_idx+1}")
                            # still return ipc-only results
                            combined_stats = ipc_stats
                        else:
                            # For each bunch in this train, sample Poisson(hpp_mu) events
                            rng = np.random.default_rng()
                            per_bunch_counts = rng.poisson(lam=float(hpp_mu), size=len(train_seeds))
                            hpp_views = []
                            for n in per_bunch_counts:
                                if n <= 0:
                                    continue
                                # sample with replacement
                                indices = rng.integers(0, num_hpp_events, size=int(n))
                                for idx in indices:
                                    hpp_views.append(_SingleEventView(htree, int(idx)))

                            # Compute HPP-only stats if any sampled
                            if hpp_views:
                                hpp_stats = analyze_detector_hits_vectorized(
                                    hpp_views, detector_name, detector_config,
                                    buffer_depths, xml_file, constants, main_xml,
                                    remove_zeros, time_cut, calo_hit_time_def, energy_thresholds,
                                    use_vectorized=use_vectorized
                                )
                                # Combined stats on merged views
                                combined_stats = analyze_detector_hits_vectorized(
                                    list(train_trees) + hpp_views, detector_name, detector_config,
                                    buffer_depths, xml_file, constants, main_xml,
                                    remove_zeros, time_cut, calo_hit_time_def, energy_thresholds,
                                    use_vectorized=use_vectorized
                                )
                            else:
                                # No HPP sampled for this train
                                combined_stats = ipc_stats
                except Exception as e:
                    print(f"Warning: Failed HPP processing for train {train_idx+1}: {e}")
                    combined_stats = ipc_stats
            else:
                combined_stats = ipc_stats

            result_bundle = {
                'ipc_only': ipc_stats,
                'hpp_only': hpp_stats,
                'combined': combined_stats
            }

            print(f"Completed train {train_idx+1}")
            return train_idx, result_bundle, None
        finally:
            for tree in train_trees:
                try:
                    tree.file.close()
                except Exception:
                    pass
        
    except Exception as e:
        error_msg = f"Error processing train {train_idx+1}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return train_idx, None, error_msg


def process_trains_parallel(trains: List[List], detector_name: str, detector_config: Any,
                          buffer_depths: List[int], xml_file: str, constants: Dict,
                          main_xml: str, base_path: str, filename_pattern: str,
                          remove_zeros: bool, time_cut: float, calo_hit_time_def: int,
                          energy_thresholds: Dict, hpp_file: Optional[str] = None,
                          hpp_mu: Optional[float] = None, max_workers: Optional[int] = None,
                          use_vectorized: bool = True) -> List[Dict]:
    """
    Process multiple trains in parallel.
    
    Parameters:
    -----------
    trains : list of lists
        List of train seeds, where each train is a list of seeds
    detector_name : str
        Name of detector to analyze
    detector_config : DetectorConfig
        Detector configuration object
    buffer_depths : list of int
        Buffer depths for analysis (counts-per-cell threshold)
    xml_file : str
        Path to detector XML file
    constants : dict
        Constants dictionary
    main_xml : str
        Path to main XML file
    base_path : str
        Base path for data files
    filename_pattern : str
        Pattern for data filenames
    remove_zeros : bool
        Whether to remove zero-position hits
    time_cut : float
        Time cut in ns
    calo_hit_time_def : int
        Calorimeter hit time definition
    energy_thresholds : dict
        Energy thresholds dictionary
    hpp_file : str, optional
        Path to HPP EDM4hep ROOT file containing many events
    hpp_mu : float, optional
        Expected number of HPP events per bunch crossing (Poisson mean)
    max_workers : int, optional
        Maximum number of worker processes
    use_vectorized : bool
        If True, enable vectorized hit processing; otherwise fall back to traditional path
        
    Returns:
    --------
    list: List of train analysis results
    """
    
    if max_workers is None:
        # For multiprocessing, use fewer workers than threads since each process uses more resources
        max_workers = min(len(trains), get_optimal_worker_count(len(trains), io_bound=False))
    
    print(f"Processing {len(trains)} trains in parallel using {max_workers} worker processes...")
    
    # Prepare arguments for each train
    train_args = []
    for train_idx, train_seeds in enumerate(trains):
        args = (
            train_idx, train_seeds, detector_name, detector_config,
            buffer_depths, xml_file, constants, main_xml,
            base_path, filename_pattern, remove_zeros, time_cut,
            calo_hit_time_def, energy_thresholds, hpp_file, hpp_mu,
            use_vectorized
        )
        train_args.append(args)
    
    # Process trains in parallel
    train_results = [None] * len(trains)  # Maintain order
    failed_trains = []
    
    start_time = time.time()
    
    # Prefer 'fork' on POSIX to avoid __main__ guard requirements; fall back to 'spawn'
    try:
        if 'fork' in mp.get_all_start_methods():
            ctx = mp.get_context('fork')
        else:
            ctx = mp.get_context('spawn')
    except Exception:
        ctx = mp.get_context('spawn')
    
    with ctx.Pool(processes=max_workers) as pool:
        try:
            # Submit all tasks
            results = pool.map(process_single_train_worker, train_args)
            
            # Process results
            for train_idx, train_stats, error in results:
                if train_stats is not None:
                    train_results[train_idx] = train_stats
                else:
                    failed_trains.append((train_idx, error))
            
        except Exception as e:
            print(f"Error in parallel train processing: {e}")
            pool.terminate()
            raise
    
    # Filter out failed trains
    successful_results = [result for result in train_results if result is not None]
    
    total_time = time.time() - start_time
    success_rate = len(successful_results) / len(trains) * 100
    
    print(f"Parallel train processing completed in {total_time:.2f}s")
    print(f"Successfully processed: {len(successful_results)}/{len(trains)} trains ({success_rate:.1f}%)")
    
    if failed_trains:
        print(f"Failed trains: {[idx+1 for idx, _ in failed_trains]}")
        for idx, error in failed_trains:
            print(f"  Train {idx+1}: {error}")
    
    return successful_results


def process_detectors_parallel(detectors_to_analyze: List[Tuple], events_trees_by_train: List,
                             trains: List[List], main_xml: str, remove_zeros: bool, 
                             time_cut: float, calo_hit_time_def: int, energy_thresholds: Dict,
                             base_path: str, filename_pattern: str, nlayer_batch: int = 1,
                             max_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Process multiple detectors in parallel using train-based analysis.
    
    Parameters:
    -----------
    detectors_to_analyze : list of tuples
        List of (detector_name, xml_file) pairs
    events_trees_by_train : list
        Pre-loaded event trees organized by train
    trains : list of lists
        List of train seeds
    main_xml : str
        Path to main XML file
    remove_zeros : bool
        Whether to remove zero-position hits
    time_cut : float
        Time cut in ns
    calo_hit_time_def : int
        Calorimeter hit time definition
    energy_thresholds : dict
        Energy thresholds dictionary
    base_path : str
        Base path for data files
    filename_pattern : str
        Pattern for data filenames
    nlayer_batch : int
        Number of layers to batch for plotting
    max_workers : int, optional
        Maximum number of worker processes
        
    Returns:
    --------
    dict: Results for all detectors
    """
    
    if max_workers is None:
        # Limit detector-level parallelism to avoid overwhelming the system
        max_workers = min(len(detectors_to_analyze), 4)
    
    print(f"Processing {len(detectors_to_analyze)} detectors in parallel...")
    
    # For now, we'll keep detector processing sequential but make train processing parallel
    # This is because detector-level parallelism requires more complex shared memory management
    detector_results = {}
    
    for detector_name, xml_file in detectors_to_analyze:
        print(f"\nProcessing detector: {detector_name}")
        
        try:
            # Import here to avoid circular imports
            from src.detector_config import get_detector_configs
            from src.geometry_parsing.k4geo_parsers import parse_detector_constants
            from src.geometry_parsing.geometry_info import get_geometry_info
            from src.hit_analysis.train_analyzer import average_train_results, plot_train_averaged_occupancy_analysis, plot_train_averaged_timing_analysis
            
            DETECTOR_CONFIGS = get_detector_configs()
            detector_config = DETECTOR_CONFIGS[detector_name]
            constants = parse_detector_constants(main_xml, detector_name, detector_xml_file=xml_file)
            geometry_info = get_geometry_info(xml_file, detector_config, constants=constants)
            
            # Define thresholds to analyze
            buffer_depths = [1, 2, 3, 4,5,6,7,8,9,10]
            
            # Process all trains for this detector in parallel
            train_results = process_trains_parallel(
                trains, detector_name, detector_config, buffer_depths,
                xml_file, constants, main_xml, base_path, filename_pattern,
                remove_zeros, time_cut, calo_hit_time_def, energy_thresholds,
                max_workers=max_workers
            )
            
            if not train_results:
                print(f"Warning: No successful train results for {detector_name}")
                continue
            
            # Average results across all trains
            stats = average_train_results(train_results)
            
            if stats is None:
                print(f"Warning: Failed to average results for {detector_name}")
                continue
            
            # Add train info to stats for plotting
            stats['train_info'] = {
                'bunches_per_train': len(trains[0]) if trains else 0,
                'num_trains': len(trains)
            }
            
            # Create visualizations with train-averaged data
            bunches_per_train = stats['train_info']['bunches_per_train']
            plot_train_averaged_occupancy_analysis(
                stats, geometry_info, 
                output_prefix=f"{detector_name}_train{bunches_per_train}",
                nlayer_batch=nlayer_batch
            )
            
            plot_train_averaged_timing_analysis(
                stats, geometry_info, 
                output_prefix=f"{detector_name}_train{bunches_per_train}"
            )
            
            detector_results[detector_name] = {
                'stats': stats,
                'geometry_info': geometry_info,
                'train_results': train_results
            }
            
            # Print summary statistics
            print(f"\nCompleted {detector_name} analysis:")
            print(f"  Processed {len(train_results)} trains successfully")
            print(f"  Total cells: {geometry_info['total_cells']}")
            
        except Exception as e:
            print(f"Error processing detector {detector_name}: {str(e)}")
            traceback.print_exc()
            continue
    
    return detector_results


def benchmark_train_parallelism(trains: List[List], detector_name: str, detector_config: Any,
                               buffer_depths: List[int], xml_file: str, constants: Dict,
                               main_xml: str, base_path: str, filename_pattern: str,
                               remove_zeros: bool, time_cut: float, calo_hit_time_def: int,
                               energy_thresholds: Dict, max_trains_to_test: int = 4) -> Dict[str, float]:
    """
    Benchmark sequential vs parallel train processing.
    
    Parameters:
    -----------
    trains : list of lists
        List of train seeds
    detector_name : str
        Detector name for testing
    detector_config : DetectorConfig
        Detector configuration
    buffer_depths : list
        Buffer depths
    xml_file : str
        XML file path
    constants : dict
        Constants dictionary
    main_xml : str
        Main XML file path
    base_path : str
        Base path for files
    filename_pattern : str
        Filename pattern
    remove_zeros : bool
        Remove zero positions flag
    time_cut : float
        Time cut value
    calo_hit_time_def : int
        Calorimeter time definition
    energy_thresholds : dict
        Energy thresholds
    max_trains_to_test : int
        Maximum number of trains to test
        
    Returns:
    --------
    dict: Benchmark results
    """
    
    test_trains = trains[:max_trains_to_test] if len(trains) > max_trains_to_test else trains
    
    print(f"Benchmarking train processing with {len(test_trains)} trains...")
    print("=" * 60)
    
    # Test sequential processing
    print("\n1. Sequential train processing:")
    print("-" * 30)
    
    from src.hit_analysis.occupancy import analyze_detector_hits
    
    start_time = time.time()
    sequential_results = []
    
    for train_idx, train_seeds in enumerate(test_trains):
        print(f"Processing train {train_idx+1}/{len(test_trains)} sequentially...")
        train_trees, failed_seeds = open_files_parallel(
            base_path, filename_pattern, train_seeds, max_workers=8
        )
        
        if train_trees:
            train_stats = analyze_detector_hits(
                train_trees, detector_name, detector_config, 
                buffer_depths, xml_file, constants, main_xml, 
                remove_zeros, time_cut, calo_hit_time_def, energy_thresholds
            )
            sequential_results.append(train_stats)
    
    sequential_time = time.time() - start_time
    print(f"Sequential time: {sequential_time:.2f} seconds")
    print(f"Sequential rate: {len(sequential_results)/sequential_time:.2f} trains/second")
    
    # Test parallel processing  
    print("\n2. Parallel train processing:")
    print("-" * 30)
    
    start_time = time.time()
    parallel_results = process_trains_parallel(
        test_trains, detector_name, detector_config, buffer_depths,
        xml_file, constants, main_xml, base_path, filename_pattern,
        remove_zeros, time_cut, calo_hit_time_def, energy_thresholds
    )
    parallel_time = time.time() - start_time
    
    print(f"Parallel time: {parallel_time:.2f} seconds")
    print(f"Parallel rate: {len(parallel_results)/parallel_time:.2f} trains/second")
    
    # Calculate speedup
    if sequential_time > 0 and parallel_time > 0:
        speedup = sequential_time / parallel_time
        print(f"\n3. Performance comparison:")
        print("-" * 30)
        print(f"Speedup factor: {speedup:.2f}x")
        print(f"Time reduction: {((sequential_time - parallel_time) / sequential_time * 100):.1f}%")
        
        if speedup > 1:
            print(f"✓ Parallel train processing is {speedup:.1f}x faster!")
        else:
            print("⚠ Parallel processing not faster (overhead may dominate)")
    
    return {
        'sequential_time': sequential_time,
        'parallel_time': parallel_time,
        'speedup': speedup if 'speedup' in locals() else 1.0,
        'sequential_results': len(sequential_results),
        'parallel_results': len(parallel_results)
    } 

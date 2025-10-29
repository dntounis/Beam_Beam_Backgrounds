#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import math
import time
import json
import importlib.util
from datetime import datetime
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from tqdm import tqdm
import matplotlib.patches as patches
import numba
from numba import jit, njit, prange
from matplotlib.transforms import Bbox

# Try importing mplhep for CMS style
try:
    import mplhep as hep
    HAS_MPLHEP = True
except ImportError:
    print("mplhep not found. Install with: pip install mplhep")
    HAS_MPLHEP = False

# Constants
ELECTRON_MASS_GEV = 0.00051099895  # GeV/c^2
ELECTRON_MASS_KG = 9.1093837e-31   # kg
ELECTRON_CHARGE = 1.60217663e-19   # C
SPEED_OF_LIGHT = 299792458         # m/s
UNITS_FACTOR = 1e3                 # m to mm conversion

# SiD Vertex Barrel layer information
VERTEX_BARREL_LAYERS = [
    # Layer 1
    {"inner_r": 13.0, "outer_r": 17.0, "z_length": 126.0, "radial_center": 15.05, "module": "Inner"},
    # Layer 2
    {"inner_r": 21.0, "outer_r": 25.0, "z_length": 126.0, "radial_center": 23.03, "module": "Outer"},
    # Layer 3
    {"inner_r": 34.0, "outer_r": 38.0, "z_length": 126.0, "radial_center": 35.79, "module": "Outer"},
    # Layer 4
    {"inner_r": 46.6, "outer_r": 50.6, "z_length": 126.0, "radial_center": 47.5, "module": "Outer"},
    # Layer 5
    {"inner_r": 59.0, "outer_r": 63.0, "z_length": 126.0, "radial_center": 59.9, "module": "Outer"}
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calculate and plot particle trajectory envelopes')
    parser.add_argument('--indir', required=True, help='Directory containing .dat files')
    parser.add_argument('--max-files', type=int, default=50, help='Maximum number of files to process')
    parser.add_argument('--field', type=float, default=5.0, help='Magnetic field strength in Tesla')
    parser.add_argument('--nz', type=int, default=400, help='Number of z bins')
    parser.add_argument('--nr', type=int, default=120, help='Number of r bins')
    parser.add_argument('--zmin', type=float, default=-300, help='Minimum z value')
    parser.add_argument('--zmax', type=float, default=300, help='Maximum z value')
    parser.add_argument('--rmin', type=float, default=-30, help='Minimum r value')
    parser.add_argument('--rmax', type=float, default=30, help='Maximum r value')
    parser.add_argument('--percentiles', type=float, nargs='+', default=[68, 95, 99, 99.9, 99.99], 
                        help='Percentiles for envelope calculation')
    parser.add_argument('--coord', choices=['x', 'y', 'r'], default='r', 
                        help='Coordinate to plot (x, y, or r)')
    parser.add_argument('--out', required=True, help='Output filename')
    parser.add_argument('--draw-2d-histo', action='store_true', help='Draw 2D histogram with envelopes')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--save-envelopes', help='Save calculated envelopes to a file')
    parser.add_argument('--plot-trajectories', type=int, default=0, 
                        help='Number of individual particle trajectories to plot (0 for none)')
    parser.add_argument('--pt-cut', type=float, default=4e-3, 
                        help='pT cut in GeV/c for particle selection')
    parser.add_argument('--roi-theta-min', type=float, default=2e-3,
                        help='Minimum theta (rad) for deflection ridge ROI')
    parser.add_argument('--roi-theta-max', type=float, default=None,
                        help='Maximum theta (rad) for deflection ridge ROI (omit for no upper bound)')
    parser.add_argument('--roi-pt-min', type=float, default=1e-3,
                        help='Minimum pT (GeV/c) for deflection ridge ROI')
    parser.add_argument('--roi-pt-max', type=float, default=None,
                        help='Maximum pT (GeV/c) for deflection ridge ROI (omit for no upper bound)')
    parser.add_argument('--smooth-envelopes', action='store_true', 
                        help='Apply smoothing to envelope curves')
    parser.add_argument('--collider', default="C³ 250 PS1", 
                        help='Collider parameter set (e.g. "C³ 250 PS1")')
    parser.add_argument('--num-bunches', type=int, default=266, 
                        help='Number of bunches')
    parser.add_argument('--detector', default="SiD_o2_v04", 
                        help='Detector configuration (e.g. "SiD_o2_v04")')
    parser.add_argument('--show-detector', action='store_true', help='Show detector barrel layers')
    parser.add_argument('--use-numba', action='store_true', help='Use Numba acceleration')
    parser.add_argument('--cache-trajectory-data', action='store_true', 
                        help='Cache trajectory calculation results to speed up repeated runs')
    parser.add_argument('--parallel-threads', type=int, default=os.cpu_count(), 
                        help='Number of parallel threads to use (default: all available cores)')
    parser.add_argument('--show-colorbar', action='store_true', 
                        help='Show colorbar for 2D histogram')
    parser.add_argument('--colorbar-min', type=float, default=0.1, 
                        help='Minimum value for colorbar scale')
    parser.add_argument('--colorbar-max', type=float, default=None, 
                        help='Maximum value for colorbar scale (omit to auto-scale to data)')
    parser.add_argument('--density-units', choices=['per_bin', 'per_mm2'], default='per_bin',
                        help='Units for the 2D histogram color map')
    parser.add_argument('--normalize-per-bunch', action='store_true',
                        help='If set, normalize aggregated histograms to one bunch crossing')
    parser.add_argument('--cmap', choices=['viridis', 'turbo', 'plasma', 'inferno', 'magma', 'root'], default='viridis',
                        help='Colormap for the 2D histogram ("root" approximates ROOT kBird)')
    parser.add_argument('--save-deflection-ridge',
                        help='Path to save deflection ridge data (JSON). Defaults to derived name next to --out')
    parser.add_argument('--no-reachability-boundary', dest='show_reachability_boundary', action='store_false',
                        default=True, help='Disable detector reach boundary overlay on pT-θ plots')
    parser.add_argument('--reachability-script',
                        help='Path to reachability_analysis.py (defaults to sibling in pair_envelopes)')
    parser.add_argument('--reachability-pt-min', type=float,
                        help='Minimum pT (GeV/c) to sample reachability boundary')
    parser.add_argument('--reachability-pt-max', type=float,
                        help='Maximum pT (GeV/c) to sample reachability boundary (defaults to plot upper bound)')
    parser.add_argument('--reachability-pt-samples', type=int, default=400,
                        help='Number of pT samples for reachability boundary')
    parser.add_argument('--reachability-linear-pt', action='store_true',
                        help='Use linear spacing for reachability boundary sampling (default is log spacing)')
    parser.add_argument('--reachability-charge', type=float, default=0.3,
                        help='Effective charge factor q for reachability boundary (GeV·T·mm)')
    parser.add_argument('--reachability-mag-field', type=float, default=5.0,
                        help='Magnetic field B0 in Tesla for reachability boundary')
    parser.add_argument('--reachability-detector-radius', type=float, default=14.0,
                        help='Detector radius in millimetres for reachability boundary')
    parser.add_argument('--reachability-z-max', type=float, default=76.0,
                        help='Detector z-extent in millimetres for reachability boundary')
    parser.add_argument('--reachability-theta-upper', type=float,
                        help='Optional theta upper bound (rad) when sampling reachability boundary')
    
    return parser.parse_args()


@njit(fastmath=True)
def pair_info_numba(energy, beta_x, beta_y, beta_z, vtx_x, vtx_y, vtx_z, 
                   electron_mass_gev, speed_of_light):
    """Numba-optimized version of pair_info"""
    # Determine charge sign (e- or e+)
    charge_sign = -1
    if energy < 0:
        charge_sign = +1
    
    gamma = abs(energy) / electron_mass_gev
    
    beta = np.sqrt(beta_x**2 + beta_y**2 + beta_z**2)
    
    # Calculate velocity components
    velocity_magnitude = np.sqrt(1 - 1/(gamma**2))  # in units of c
    velocity_x = beta_x / beta * velocity_magnitude
    velocity_y = beta_y / beta * velocity_magnitude
    velocity_z = beta_z / beta * velocity_magnitude
    
    # Convert to SI units
    x = vtx_x * 1e-9  # nm to m
    y = vtx_y * 1e-9
    z = vtx_z * 1e-9
    
    vx = velocity_x * speed_of_light  # m/s
    vy = velocity_y * speed_of_light
    vz = velocity_z * speed_of_light
    
    # Calculate transverse momentum
    pT = gamma * electron_mass_gev * np.sqrt(velocity_x**2 + velocity_y**2)  # GeV/c
    
    return x, y, z, charge_sign, vx, vy, vz, gamma, pT


def pair_info(energy, beta_x, beta_y, beta_z, vtx_x, vtx_y, vtx_z):
    """
    Process information about a particle.
    
    Args:
        energy: Energy of the particle (GeV)
        beta_x, beta_y, beta_z: Components of the velocity vector (in units of c)
        vtx_x, vtx_y, vtx_z: Components of the vertex position (nm)
        
    Returns:
        tuple: (x, y, z, charge_sign, vx, vy, vz, gamma, pT)
    """
    return pair_info_numba(energy, beta_x, beta_y, beta_z, vtx_x, vtx_y, vtx_z,
                         ELECTRON_MASS_GEV, SPEED_OF_LIGHT)


@njit(fastmath=True)
def calculate_trajectory_numba(x0, y0, z0, charge_sign, v0x, v0y, v0z, gamma, magnetic_field, 
                             gyrofrequency, gyroperiod, z_min, z_max, units_factor):
    """Numba-optimized trajectory calculation"""
    # Calculate radius and phase
    transverse_velocity = np.sqrt(v0x**2 + v0y**2)
    if transverse_velocity < 1e-10:  # Handle numerical issues with very small velocities
        radius = 0.0
        phi = 0.0
    else:
        radius = transverse_velocity / abs(gyrofrequency)  # m
        phi = np.arctan2(-v0y, v0x)  # Use arctan2 for correct quadrant
    
    # Calculate center of circular motion
    x_center = x0 - radius * np.sin(phi)
    y_center = y0 - radius * np.cos(phi)
    
    # Calculate helix pitch in mm
    pitch = abs(v0z) * gyroperiod * units_factor  # mm
    
    # Determine number of periods and points per period
    if abs(v0z) < 1e-10:  # Handle case of near-zero longitudinal velocity
        n_periods = 1
        n_points = 20
    else:
        z_range = z_max - z_min  # mm
        n_periods = max(1, math.ceil(z_range / pitch) if pitch > 0 else 1)
        n_points = max(8, math.ceil(pitch / 1.5)) if pitch > 0 else 20
    
    # Limit total points to avoid memory issues
    max_points = 8000  # Lower than original to save memory
    if n_periods * n_points > max_points:
        if math.floor(max_points / n_periods) > 0:
            n_points = math.floor(max_points / n_periods)
        else:
            n_points = 1
            n_periods = max_points
    
    # Create time array
    n_total_points = n_periods * n_points + 1
    t_values = np.linspace(0, n_periods * gyroperiod, n_total_points)
    
    # Pre-allocate arrays
    x = np.zeros(n_total_points)
    y = np.zeros(n_total_points)
    z = np.zeros(n_total_points)
    
    # Calculate trajectory
    for i in range(n_total_points):
        t = t_values[i]
        x[i] = (radius * np.sin(gyrofrequency * t + phi) + x_center) * units_factor
        y[i] = (radius * np.cos(gyrofrequency * t + phi) + y_center) * units_factor
        z[i] = (z0 + v0z * t) * units_factor
    
    # Count points within z range
    count = 0
    for i in range(n_total_points):
        if z_min <= z[i] <= z_max:
            count += 1
    
    if count == 0:
        # No points in range, return empty arrays
        return np.zeros(1), np.zeros(1), np.zeros(1)
    
    # Create filtered arrays
    x_filtered = np.zeros(count)
    y_filtered = np.zeros(count)
    z_filtered = np.zeros(count)
    
    # Fill filtered arrays
    j = 0
    for i in range(n_total_points):
        if z_min <= z[i] <= z_max:
            x_filtered[j] = x[i]
            y_filtered[j] = y[i]
            z_filtered[j] = z[i]
            j += 1
    
    return x_filtered, y_filtered, z_filtered


def calculate_trajectory(x0, y0, z0, charge_sign, v0x, v0y, v0z, gamma, magnetic_field, z_min, z_max):
    """
    Calculate particle trajectory in magnetic field.
    
    Args:
        x0, y0, z0: Initial position (m)
        charge_sign: Charge sign (-1 for electron, +1 for positron)
        v0x, v0y, v0z: Initial velocity (m/s)
        gamma: Lorentz factor
        magnetic_field: Magnetic field strength (T)
        z_min, z_max: Z range for tracking (mm)
        
    Returns:
        tuple: (x, y, z) arrays of trajectory points in mm
    """
    # Calculate gyration parameters
    gyrofrequency = charge_sign * ELECTRON_CHARGE * magnetic_field / (gamma * ELECTRON_MASS_KG)  # s^-1
    gyroperiod = 2 * np.pi / abs(gyrofrequency)  # s
    
    # Call Numba-optimized function
    return calculate_trajectory_numba(
        x0, y0, z0, charge_sign, v0x, v0y, v0z, gamma, magnetic_field,
        gyrofrequency, gyroperiod, z_min, z_max, UNITS_FACTOR
    )


@njit(fastmath=True)
def safe_hist2d_numba(z_data, r_data, z_edges, r_edges):
    """Numba-optimized histogram calculation"""
    n_z_bins = len(z_edges) - 1
    n_r_bins = len(r_edges) - 1
    hist = np.zeros((n_z_bins, n_r_bins))
    
    for i in range(len(z_data)):
        z = z_data[i]
        r = r_data[i]
        
        # Find z bin
        z_bin = -1
        for j in range(n_z_bins):
            if z_edges[j] <= z < z_edges[j+1]:
                z_bin = j
                break
        
        # Find r bin
        r_bin = -1
        for j in range(n_r_bins):
            if r_edges[j] <= r < r_edges[j+1]:
                r_bin = j
                break
        
        # Update histogram if valid bin found
        if z_bin >= 0 and r_bin >= 0:
            hist[z_bin, r_bin] += 1
    
    return hist


@njit(fastmath=True)
def calculate_percentile_numba(histogram_slice, r_centers, percentile):
    """Numba-optimized percentile calculation for a single z slice"""
    # Find indices for positive and negative r values
    positive_indices = np.zeros(len(r_centers), dtype=np.int32)
    negative_indices = np.zeros(len(r_centers), dtype=np.int32)
    n_pos = 0
    n_neg = 0
    
    for i in range(len(r_centers)):
        if r_centers[i] >= 0:
            positive_indices[n_pos] = i
            n_pos += 1
        if r_centers[i] <= 0:
            negative_indices[n_neg] = i
            n_neg += 1
    
    positive_indices = positive_indices[:n_pos]
    negative_indices = negative_indices[:n_neg]
    
    # Get the distributions
    hist_pos = np.zeros(n_pos)
    hist_neg = np.zeros(n_neg)
    
    for i in range(n_pos):
        hist_pos[i] = histogram_slice[positive_indices[i]]
    
    for i in range(n_neg):
        hist_neg[i] = histogram_slice[negative_indices[i]]
    
    # Skip if no data
    if np.sum(hist_pos) == 0 or np.sum(hist_neg) == 0:
        return 0.0, 0.0
    
    # Calculate cumulative distributions
    cum_pos = np.zeros(n_pos)
    cum_pos[0] = hist_pos[0]
    for i in range(1, n_pos):
        cum_pos[i] = cum_pos[i-1] + hist_pos[i]
    cum_pos = cum_pos / cum_pos[-1] * 100
    
    cum_neg = np.zeros(n_neg)
    cum_neg[n_neg-1] = hist_neg[n_neg-1]
    for i in range(n_neg-2, -1, -1):
        cum_neg[i] = cum_neg[i+1] + hist_neg[i]
    cum_neg = cum_neg / cum_neg[0] * 100
    
    # Find percentile values
    envelope_pos = 0.0
    envelope_neg = 0.0
    
    # For positive r values
    for i in range(n_pos):
        if cum_pos[i] >= percentile:
            envelope_pos = r_centers[positive_indices[i]]
            break
    
    # For negative r values
    for i in range(n_neg-1, -1, -1):
        if cum_neg[i] >= percentile:
            envelope_neg = r_centers[negative_indices[i]]
            break
    
    return envelope_pos, envelope_neg

def combine_histograms(hist_list):
    """
    Combine histogram data from multiple files.
    
    Args:
        hist_list: List of histogram dictionaries
        
    Returns:
        dict: Combined histogram dictionary
    """
    if not hist_list:
        return None
    
    # Initialize combined dictionary
    combined = {
        'r_vs_z': np.zeros_like(hist_list[0]['r_vs_z']),
        'x_vs_z': np.zeros_like(hist_list[0]['x_vs_z']),
        'y_vs_z': np.zeros_like(hist_list[0]['y_vs_z']),
        'r_plus_62': [],
        'r_minus_62': [],
        'r_max': [],
        'pt_values': [],
        'theta_values': []
    }
    
    # Add histograms and lists
    for hist in hist_list:
        combined['r_vs_z'] += hist['r_vs_z']
        combined['x_vs_z'] += hist['x_vs_z']
        combined['y_vs_z'] += hist['y_vs_z']
        combined['r_plus_62'].extend(hist['r_plus_62'])
        combined['r_minus_62'].extend(hist['r_minus_62'])
        combined['r_max'].extend(hist['r_max'])
        if 'pt_values' in hist:
            combined['pt_values'].extend(hist['pt_values'])
        if 'theta_values' in hist:
            combined['theta_values'].extend(hist['theta_values'])
    
    return combined


def process_files_parallel(file_list, magnetic_field, nz, nr, z_min, z_max, r_min, r_max, pt_cut, keep_trajectories, num_threads=None):
    """
    Process multiple .dat files in parallel.
    
    Args:
        file_list: List of .dat files to process
        magnetic_field: Magnetic field strength in Tesla
        nz, nr: Number of bins in z and r directions
        z_min, z_max: Z range for analysis
        pt_cut: pT cut for selection
        keep_trajectories: Whether to keep trajectory data for plotting
        num_threads: Number of parallel threads to use (default: all available cores)
        
    Returns:
        tuple: Combined histograms, trajectories, and statistics
    """
    results = []
    
    # Use specified number of threads or default to all available
    max_workers = num_threads if num_threads is not None else os.cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Print information about parallel processing
        print(f"Processing {len(file_list)} files using {max_workers} parallel workers")
        
        # Submit tasks
        futures = [
            executor.submit(process_dat_file, fname, magnetic_field, nz, nr, z_min, z_max, r_min, r_max, pt_cut, keep_trajectories)
            for fname in file_list
        ]
        
        # Process results as they complete
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                print(f"Completed file {i+1}/{len(file_list)}")
            except Exception as e:
                print(f"Error in parallel processing: {e}")
    
    # Combine results
    hist_dicts = [r[0] for r in results]
    all_trajectories = []
    total_pass = 0
    total_particles = 0
    
    for _, trajectories, pass_count, total_count in results:
        all_trajectories.extend(trajectories)
        total_pass += pass_count
        total_particles += total_count
    
    combined_hist = combine_histograms(hist_dicts)
    
    return combined_hist, all_trajectories, total_pass, total_particles

def process_particle_trajectory(particle_data, magnetic_field, nz, nr, z_min, z_max, r_min, r_max, pt_cut, 
                               r_vs_z_hist, x_vs_z_hist, y_vs_z_hist,
                               r_hist_plus_62, r_hist_minus_62, r_max_hist):
    """
    Process a single particle's trajectory in the magnetic field.
    
    Args:
        particle_data: Tuple of particle parameters
        magnetic_field: Magnetic field strength in Tesla
        nz, nr: Number of bins in z and r directions
        z_min, z_max: Z range for analysis
        pt_cut: pT cut for selection
        r_vs_z_hist, x_vs_z_hist, y_vs_z_hist: Histograms to fill
        r_hist_plus_62, r_hist_minus_62, r_max_hist: Lists to fill
        
    Returns:
        tuple: (trajectory, pass_selection, pT_value, theta_value)
            trajectory: (x, y, z, r) if particles passes selection, else None
            pass_selection: Boolean indicating if particle passes selection
            pT_value: Transverse momentum (GeV/c)
            theta_value: Polar angle with respect to the z axis (rad)
    """
    # Extract particle parameters (using only first 7 columns)
    energy, beta_x, beta_y, beta_z, vtx_x, vtx_y, vtx_z = particle_data[:7]
    
    # Calculate particle parameters
    x0, y0, z0, charge_sign, v0x, v0y, v0z, gamma, pT = pair_info(energy, beta_x, beta_y, beta_z, vtx_x, vtx_y, vtx_z)

    pZ = gamma * ELECTRON_MASS_GEV * (v0z / SPEED_OF_LIGHT)
    #theta = np.arctan2(pT, pZ)
    #Dimitris - test: force theta between 0 and pi/2
    theta = np.arctan2(pT, np.abs(pZ))

    # Check if particle passes pT cut immediately
    pass_selection = pT > pt_cut
    
    # Calculate trajectory
    try:
        x, y, z = calculate_trajectory(x0, y0, z0, charge_sign, v0x, v0y, v0z, gamma, 
                                     magnetic_field, z_min, z_max)
    except Exception as e:
        # If trajectory calculation fails, skip this particle
        return None, False, pT, theta
    
    # Skip if no valid trajectory points
    if len(z) == 0 or len(x) == 0 or len(y) == 0:
        return None, pass_selection, pT, theta
    
    # Calculate r with sign (positive if y>0, negative if y<0)
    r = np.sqrt(x**2 + y**2) * np.sign(y)
    
    # Calculate bin edges
    z_bin_edges = np.linspace(z_min, z_max, nz + 1)
    r_bin_edges = np.linspace(r_min, r_max, nr + 1)
    
    # Use Numba-optimized histogram function
    r_vs_z_hist_update = safe_hist2d_numba(z, r, z_bin_edges, r_bin_edges)
    x_vs_z_hist_update = safe_hist2d_numba(z, x * np.sign(y), z_bin_edges, r_bin_edges)
    y_vs_z_hist_update = safe_hist2d_numba(z, y * np.sign(x), z_bin_edges, r_bin_edges)
    
    # Update histograms
    r_vs_z_hist += r_vs_z_hist_update
    x_vs_z_hist += x_vs_z_hist_update
    y_vs_z_hist += y_vs_z_hist_update
    
    # Record r values at z = ±62mm
    if v0z > 0:
        idx = np.argmin(np.abs(z - 62))
        if idx < len(z) and abs(z[idx] - 62) <= 1.5:
            r_hist_plus_62.append(r[idx])
    else:
        idx = np.argmin(np.abs(z + 62))
        if idx < len(z) and abs(z[idx] + 62) <= 1.5:
            r_hist_minus_62.append(r[idx])
    
    # Record maximum r within |z| < 62.5mm
    z_mask = np.abs(z) <= 62.5
    if np.any(z_mask):
        r_abs = np.abs(r[z_mask])
        if len(r_abs) > 0:
            idx_max = np.argmax(r_abs)
            r_max_hist.append(z[z_mask][idx_max])
    
    # Return trajectory if particle passes selection
    if pass_selection:
        return (x, y, z, r), True, pT, theta
    else:
        return None, False, pT, theta


def process_dat_file(filename, magnetic_field, nz, nr, z_min, z_max, r_min, r_max, pt_cut, keep_trajectories=False):
    """
    Process a single .dat file of particle data.
    
    Args:
        filename: Path to the .dat file
        magnetic_field: Magnetic field strength in Tesla
        nz, nr: Number of bins in z and r directions
        z_min, z_max: Z range for analysis
        pt_cut: pT cut for selection
        keep_trajectories: Whether to keep trajectory data for plotting
        
    Returns:
        tuple: (histograms, trajectories, pass_count, total_count)
    """
    # Initialize histograms and lists
    r_vs_z_hist = np.zeros((nz, nr))
    x_vs_z_hist = np.zeros((nz, nr))
    y_vs_z_hist = np.zeros((nz, nr))
    
    r_hist_plus_62 = []
    r_hist_minus_62 = []
    r_max_hist = []
    pt_values = []
    theta_values = []
    
    trajectories = []
    pass_count = 0
    total_count = 0
    
    # Process the file
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
            # Pre-process data to avoid Python overhead in the particle processing loop
            particle_data_list = []
            for line in lines:
                try:
                    # Parse line - handle any number of columns
                    data = [float(x) for x in line.strip().split()]
                    if len(data) >= 7:
                        particle_data_list.append(data[:7])  # Just keep the first 7 columns
                except Exception:
                    continue
            
            total_count = len(particle_data_list)
            
            # Process particles in batches
            batch_size = 100
            for i in tqdm(range(0, total_count, batch_size), 
                          desc=f"Processing {os.path.basename(filename)}",
                          disable=total_count < 1000):
                batch = particle_data_list[i:min(i+batch_size, total_count)]
                
                for particle_data in batch:
                    # Process particle
                    try:
                        trajectory, passes, pT_value, theta_value = process_particle_trajectory(
                            particle_data, magnetic_field, nz, nr, z_min, z_max, r_min, r_max, pt_cut,
                            r_vs_z_hist, x_vs_z_hist, y_vs_z_hist,
                            r_hist_plus_62, r_hist_minus_62, r_max_hist
                        )

                        if np.isfinite(pT_value) and np.isfinite(theta_value):
                            pt_values.append(pT_value)
                            theta_values.append(theta_value)
                        
                        if passes:
                            pass_count += 1
                            if keep_trajectories and len(trajectories) < 100:
                                trajectories.append(trajectory)
                    except Exception as e:
                        print(f"Error processing particle: {e}")
                        continue
    
    except Exception as e:
        print(f"Error opening file {filename}: {e}")
    
    return {
        'r_vs_z': r_vs_z_hist,
        'x_vs_z': x_vs_z_hist,
        'y_vs_z': y_vs_z_hist,
        'r_plus_62': r_hist_plus_62,
        'r_minus_62': r_hist_minus_62,
        'r_max': r_max_hist,
        'pt_values': pt_values,
        'theta_values': theta_values,
    }, trajectories, pass_count, total_count


def calculate_envelopes(histogram, percentiles, z_min, z_max, smooth=False):
    """
    Calculate envelope curves for given percentiles.
    
    Args:
        histogram: 2D histogram (z vs r/x/y)
        percentiles: List of percentile values to calculate
        z_min, z_max: Z range
        smooth: Whether to apply smoothing to the envelopes
        
    Returns:
        dict: Dictionary of envelope curves keyed by percentile
    """
    nz, nr = histogram.shape
    
    # Calculate bin centers for z
    z_edges = np.linspace(z_min, z_max, nz + 1)
    z_centers = 0.5 * (z_edges[1:] + z_edges[:-1])
    
    # Calculate bin centers for r
    r_edges = np.linspace(-30, 30, nr + 1)
    r_centers = 0.5 * (r_edges[1:] + r_edges[:-1])
    
    envelopes = {}
    
    for percentile in percentiles:
        # Initialize envelope arrays
        envelope_positive = np.zeros(nz)
        envelope_negative = np.zeros(nz)
        
        # Process each z bin
        for i in range(nz):
            # Skip empty bins
            if np.sum(histogram[i, :]) == 0:
                continue
            
            # Use Numba-optimized percentile calculation
            pos, neg = calculate_percentile_numba(histogram[i, :], r_centers, percentile)
            envelope_positive[i] = pos
            envelope_negative[i] = neg
        
        # Apply smoothing if requested
        if smooth:
            window_size = 5
            kernel = np.ones(window_size) / window_size
            
            # Apply convolution for smoothing
            envelope_positive_smooth = np.convolve(envelope_positive, kernel, mode='same')
            envelope_negative_smooth = np.convolve(envelope_negative, kernel, mode='same')
            
            # Keep original values at the edges
            envelope_positive_smooth[:window_size//2] = envelope_positive[:window_size//2]
            envelope_positive_smooth[-window_size//2:] = envelope_positive[-window_size//2:]
            envelope_negative_smooth[:window_size//2] = envelope_negative[:window_size//2]
            envelope_negative_smooth[-window_size//2:] = envelope_negative[-window_size//2:]
            
            envelope_positive = envelope_positive_smooth
            envelope_negative = envelope_negative_smooth
        
        envelopes[percentile] = (z_centers, envelope_positive, envelope_negative)
    
    return envelopes


def _resolve_reachability_path(custom_path):
    if custom_path:
        return custom_path
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(this_dir, 'reachability_analysis.py'),
        os.path.join(os.path.dirname(this_dir), 'reachability_analysis.py'),
        os.path.join(os.path.dirname(os.path.dirname(this_dir)), 'reachability_analysis.py'),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _load_reachability_module(script_path):
    try:
        spec = importlib.util.spec_from_file_location('reachability_analysis_runtime', script_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        print(f"Warning: failed to load reachability module '{script_path}': {exc}")
        return None


def _sample_reachability_boundary(args, pt_upper_hint):
    script_path = _resolve_reachability_path(getattr(args, 'reachability_script', None))
    if not script_path:
        print('Warning: reachability_analysis.py not found; skipping detector reach boundary overlay.')
        return None, None
    print(f"Reachability script path: {script_path}")

    module = _load_reachability_module(script_path)
    if module is None:
        return None, None
    print("Reachability module loaded successfully.")

    compute_boundary = getattr(module, 'compute_boundary', None)
    if compute_boundary is None:
        print(f"Warning: reachability module '{script_path}' has no compute_boundary(); skipping overlay.")
        return None, None

    charge = getattr(args, 'reachability_charge', None)
    mag_field = getattr(args, 'reachability_mag_field', None)
    r_det = getattr(args, 'reachability_detector_radius', None)
    z_max = getattr(args, 'reachability_z_max', None)

    if (charge is None or charge <= 0 or mag_field is None or mag_field <= 0
            or r_det is None or r_det <= 0 or z_max is None or z_max <= 0):
        print('Warning: invalid reachability parameters; skipping boundary overlay.')
        return None, None

    pt_min_condition = (charge * mag_field * r_det) / 2000.0
    provided_pt_min = getattr(args, 'reachability_pt_min', None)
    pt_min = max(provided_pt_min if provided_pt_min is not None else pt_min_condition, pt_min_condition)
    print(f"Reachability sampling pT_min condition: {pt_min_condition:.6e} GeV/c, using pT_min={pt_min:.6e} GeV/c")

    axis_pt_max = pt_upper_hint if (pt_upper_hint is not None and np.isfinite(pt_upper_hint)) else None
    provided_pt_max = getattr(args, 'reachability_pt_max', None)
    pt_max = provided_pt_max if provided_pt_max is not None else axis_pt_max
    if pt_max is None or not np.isfinite(pt_max) or pt_max <= pt_min:
        pt_max = max(pt_min * 1.05, pt_min + 1e-3)
    print(f"Reachability sampling pT_max={pt_max:.6e} GeV/c (axis hint={axis_pt_max})")

    samples = max(int(getattr(args, 'reachability_pt_samples', 400)), 5)
    if getattr(args, 'reachability_linear_pt', False):
        pt_values = np.linspace(pt_min, pt_max, samples)
    else:
        pt_values = np.logspace(np.log10(pt_min), np.log10(pt_max), samples)
    if pt_values.size > 0:
        pt_values[0] = pt_min
        pt_values[-1] = pt_max
    print(f"Reachability sampling {samples} points (log spacing={not getattr(args, 'reachability_linear_pt', False)})")

    theta_upper = getattr(args, 'reachability_theta_upper', None)
    try:
        boundary_result = compute_boundary(
            pt_values=pt_values,
            q=charge,
            B0=mag_field,
            r_det=r_det,
            z_max=z_max,
            theta_upper=theta_upper,
        )
    except Exception as exc:
        print(f"Warning: failed to compute reachability boundary: {exc}")
        return None, None

    if isinstance(boundary_result, (tuple, list)) and len(boundary_result) == 2:
        theta_vals, pt_boundary = boundary_result
    else:
        theta_vals = boundary_result
        pt_boundary = pt_values

    theta_vals = np.asarray(theta_vals, dtype=float)
    pt_boundary = np.asarray(pt_boundary, dtype=float)
    valid = np.isfinite(theta_vals) & np.isfinite(pt_boundary) & (theta_vals > 0) & (pt_boundary > 0)
    if not np.any(valid):
        print('Warning: no valid reachability boundary points computed; skipping overlay.')
        return None, None

    theta_vals = theta_vals[valid]
    pt_boundary = pt_boundary[valid]
    order = np.argsort(theta_vals)
    theta_vals = theta_vals[order]
    pt_boundary = pt_boundary[order]
    return theta_vals, pt_boundary

def plot_detector_layers(ax, detector_color='blue', coord='r'):
    """
    Add detector silicon barrel layers to the plot.
    
    Args:
        ax: Matplotlib axes object
        detector_color: Color to use for the detector layers
        coord: Coordinate type ('x', 'y', or 'r')
    
    Returns:
        list: List of (patch, label) tuples for the legend
    """
    legend_elements = []
    
    # Only proceed if we're plotting r vs z
    if coord != 'r':
        return legend_elements
    
    # Add each barrel layer
    for i, layer in enumerate(VERTEX_BARREL_LAYERS):
        # Top half (positive y/r)
        z_half = layer["z_length"] / 2
        
        # Add rectangle patches for positive and negative r
        rect_pos = patches.Rectangle(
            (-z_half, layer["inner_r"]), 
            layer["z_length"], 
            layer["outer_r"] - layer["inner_r"], 
            linewidth=1, 
            edgecolor=detector_color, 
            facecolor=detector_color,
            alpha=0.3,
            zorder=1  # Ensure it's behind other elements
        )
        
        rect_neg = patches.Rectangle(
            (-z_half, -layer["outer_r"]), 
            layer["z_length"], 
            layer["outer_r"] - layer["inner_r"], 
            linewidth=1, 
            edgecolor=detector_color, 
            facecolor=detector_color,
            alpha=0.3,
            zorder=1
        )
        
        # Add to the plot
        ax.add_patch(rect_pos)
        ax.add_patch(rect_neg)
        
        # Add to legend only once (for the first layer)
        if i == 0:
            legend_elements.append((rect_pos, f"Silicon Vertex Barrel"))
    
    return legend_elements


def plot_envelopes(envelopes, histogram=None, trajectories=None, 
                   coord='r', z_min=-300, z_max=300, 
                   draw_2d_histo=False, output_file=None,
                   percentiles=None, collider="C³ 250 PS1", 
                   num_bunches=266, detector="SiD_o2_v04", field=5.0, 
                   show_detector=True, show_colorbar=False,
                   colorbar_min=0.1, colorbar_max=2e4,
                   r_min=-40, r_max=40,
                   density_units='per_bin', cmap_name='viridis'):
    """
    Plot envelopes, 2D histogram, and optionally trajectories with CMS style.
    
    Args:
        envelopes: Dictionary of envelope curves keyed by percentile
        histogram: 2D histogram to plot as background
        trajectories: List of trajectories to plot
        coord: Coordinate type ('x', 'y', or 'r')
        z_min, z_max: Z range for plot
        draw_2d_histo: Whether to draw 2D histogram
        output_file: Path to save the plot
        percentiles: List of percentiles to include in the plot
        collider: Collider parameter set name
        num_bunches: Number of bunches
        detector: Detector configuration
        field: Magnetic field strength in Tesla
        show_detector: Whether to show detector barrel layers
        show_colorbar: Whether to show colorbar for 2D histogram
        colorbar_min: Minimum value for colorbar scale
        colorbar_max: Maximum value for colorbar scale
    """
    # Apply CMS style if mplhep is available
    if HAS_MPLHEP:
        hep.style.use("CMS")
    
    # Create figure
    if show_colorbar:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
    colorbar_ax = None

    # Define color map for percentiles
    colors = {
        68: 'black',
        95: 'red',
        99: 'lime',
        99.9: 'blue',
        99.97: 'magenta',
        99.99: 'magenta',
        'detector': 'blue'
    }
    
    # Add detector layers if requested (behind everything else)
    detector_legend_elements = []
    if show_detector and coord == 'r':
        detector_legend_elements = plot_detector_layers(ax, colors['detector'], coord)
   
    # Plot 2D histogram if requested
    if draw_2d_histo and histogram is not None:
        # Compute per-bin area if density scaling is requested
        # histogram has shape (nz, nr) with r in [-30, 30]
        nz, nr = histogram.shape
        dz = (z_max - z_min) / nz
        dr = (r_max - r_min) / nr
        area = dz * dr  # mm^2 per bin

        if density_units == 'per_mm2':
            hist_for_plot = histogram / area
            colorbar_label = 'Tracks/mm²'
        else:
            hist_for_plot = histogram
            colorbar_label = 'Tracks/bin'

        # Mask bins at or below the colorbar minimum so they render as white
        threshold = max(colorbar_min, 0)
        masked_hist = np.ma.masked_where(hist_for_plot.T < threshold, hist_for_plot.T)
        # Select colormap (optionally exact ROOT kBird using provided stops & RGB)
        if cmap_name == 'root':
            stops = [0.0000, 0.1250, 0.2500, 0.3750, 0.5000, 0.6250, 0.7500, 0.8750, 1.0000]
            red   = [0.2082, 0.0592, 0.0780, 0.0232, 0.1802, 0.5301, 0.8186, 0.9956, 0.9764]
            green = [0.1664, 0.3599, 0.5041, 0.6419, 0.7178, 0.7492, 0.7328, 0.7862, 0.9832]
            blue  = [0.5293, 0.8684, 0.8385, 0.7914, 0.6425, 0.4662, 0.3499, 0.1968, 0.0539]
            color_points = [(stops[i], (red[i], green[i], blue[i])) for i in range(len(stops))]
            cmap = LinearSegmentedColormap.from_list('root_kbird', color_points, N=255)
        else:
            cmap = plt.cm.get_cmap(cmap_name).copy()
        cmap.set_bad(color='white')
        # Build normalization (auto-scale vmax if not provided)
        norm = LogNorm(vmin=colorbar_min) if colorbar_max is None else LogNorm(vmin=colorbar_min, vmax=colorbar_max)
        im = ax.imshow(
            masked_hist,
            extent=[z_min, z_max, r_min, r_max],
            origin='lower',
            aspect='auto',
            norm=norm,
            cmap=cmap
        )
        
        # Add colorbar if requested (in a fixed side axis that does not shrink the main axes)
        if show_colorbar:
            pos = ax.get_position()
            gap = -0.033 #0.01
            cbar_width = 0.018 #0.02
            cax = fig.add_axes([pos.x1 + gap, pos.y0, cbar_width, pos.height])
            colorbar_ax = cax
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(colorbar_label, fontsize=17)

    # Plot individual trajectories if provided
    if trajectories and not draw_2d_histo:
        for traj in trajectories[:20]:  # Limit to prevent overcrowding
            x, y, z, r = traj
            if coord == 'r':
                ax.plot(z, r, color='blue', alpha=0.1, linewidth=0.5)
            elif coord == 'x':
                ax.plot(z, x * np.sign(y), color='blue', alpha=0.1, linewidth=0.5)
            elif coord == 'y':
                ax.plot(z, y * np.sign(x), color='blue', alpha=0.1, linewidth=0.5)
    
    # Create a list to store handles for legend entries
    legend_handles = []
    legend_labels = []
    
    # Plot envelopes
    for percentile, (z, pos, neg) in envelopes.items():
        if percentiles and percentile not in percentiles:
            continue
            
        color = colors.get(percentile, 'gray')
        label = f'{percentile:.2f}%'
        line, = ax.plot(z, pos, color=color, linewidth=2)
        ax.plot(z, neg, color=color, linewidth=2)
        
        # Add to legend
        legend_handles.append(line)
        legend_labels.append(label)
    
    # Draw beampipe/detector boundaries (with a label for the legend)
    boundary_kwargs = {'color': 'black', 'linewidth': 4, 'alpha': 0.28}
    beampipe_line, = ax.plot([-62, 62], [12, 12], **boundary_kwargs)
    ax.plot([62, 200], [12, 21], **boundary_kwargs)
    ax.plot([200, 300], [21, 29], **boundary_kwargs)
    ax.plot([-62, -200], [12, 21], **boundary_kwargs)
    ax.plot([-200, -300], [21, 29], **boundary_kwargs)
    
    ax.plot([-62, 62], [-12, -12], **boundary_kwargs)
    ax.plot([62, 200], [-12, -21], **boundary_kwargs)
    ax.plot([200, 300], [-21, -29], **boundary_kwargs)
    ax.plot([-62, -200], [-12, -21], **boundary_kwargs)
    ax.plot([-200, -300], [-21, -29], **boundary_kwargs)
    
    # Add beampipe to legend
    legend_handles.append(beampipe_line)
    legend_labels.append('Beampipe boundary')
    
    # Add detector layers to legend if present
    for patch, label in detector_legend_elements:
        legend_handles.append(patch)
        legend_labels.append(label)

    # Set labels and limits
    ax.set_xlim(z_min, z_max)
    ax.set_ylim(r_min, r_max)
    ax.set_xlabel('z [mm]', fontsize=18)
    ax.set_ylabel(f'{coord} [mm]', fontsize=18)
    
    # Create the legend with all elements
    # Position it outside the main plot on the right
    legend_obj = None
    legend_kwargs = {
        'fontsize': 16,
        'frameon': True,
        'fancybox': True,
        'shadow': True
    }

    legend_anchor = (1.02, 1.0)
    legend_transform = ax.transAxes
    if colorbar_ax is not None:
        cbar_pos = colorbar_ax.get_position()
        # Move legend to the right of the colorbar with a fixed gap in figure coords
        legend_anchor = (cbar_pos.x1 + 0.06, cbar_pos.y1)
        legend_transform = fig.transFigure
    elif show_colorbar:
        # Colorbar was requested but not drawn; leave a little extra gap just in case
        legend_anchor = (1.12, 1.0)

    legend_obj = ax.legend(
        legend_handles,
        legend_labels,
        loc='upper left',
        bbox_to_anchor=legend_anchor,
        bbox_transform=legend_transform,
        **legend_kwargs
    )     
    
    # Reserve fixed margins so the axes area remains identical across
    # the saved versions (with and without legend) and avoid tight_layout
    # which can clip colorbar labels.
    
    # Add collider and bunch information at the top
    if num_bunches is None:
        bunch_count = 'N/A'
        bunch_text = 'bunches'
    else:
        bunch_text = 'bunch' if num_bunches == 1 else 'bunches'
        bunch_count = f"{num_bunches}"

    ax.set_title(
        f"{collider} ({bunch_count} {bunch_text})",
        fontsize=19)

    # Add detector and field information
    detector_text = f'{detector} (B={field}T)'
    ax.text(0.99, 1.04, detector_text, transform=ax.transAxes, 
            fontsize=19, ha='right', va='top')
        
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(left=0.115, right=0.85, bottom=0.11, top=0.95)

    def _collect_tight_bboxes(renderer):
        bounding_boxes = []
        for artist in (ax, colorbar_ax):
            if artist is None:
                continue
            bbox = artist.get_tightbbox(renderer)
            if bbox is not None:
                bounding_boxes.append(bbox.transformed(fig.dpi_scale_trans.inverted()))
        if bounding_boxes:
            return Bbox.union(bounding_boxes)
        return None
    
    # Save or show plot
    if output_file:
        pad_inches = 0.02

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        base_bbox = _collect_tight_bboxes(renderer)

        if legend_obj is not None:
            legend_bbox = legend_obj.get_window_extent(renderer)
            legend_bbox_inches = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
            if base_bbox is not None:
                bbox_with_legend = Bbox.union([base_bbox, legend_bbox_inches])
            else:
                bbox_with_legend = legend_bbox_inches

            fig.savefig(output_file, dpi=300, bbox_inches=bbox_with_legend, pad_inches=pad_inches)
            print(f"Plot saved to {output_file}")

            legend_obj.remove()
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            base_bbox = _collect_tight_bboxes(renderer)

            output_no_legend = _build_additional_output_path(output_file, '_no_legend')
            if base_bbox is not None:
                fig.savefig(output_no_legend, dpi=300, bbox_inches=base_bbox, pad_inches=pad_inches)
            else:
                fig.savefig(output_no_legend, dpi=300)
            print(f"Plot saved without legend to {output_no_legend}")
        else:
            if base_bbox is not None:
                fig.savefig(output_file, dpi=300, bbox_inches=base_bbox, pad_inches=pad_inches)
            else:
                fig.savefig(output_file, dpi=300)
            print(f"Plot saved to {output_file}")
    else:
        plt.show()


def _build_additional_output_path(base_output, suffix, forced_ext=None):
    """Create an output path by inserting a suffix before the extension."""
    root, ext = os.path.splitext(base_output)
    if forced_ext:
        ext = forced_ext
    elif not ext:
        ext = '.png'
    return f"{root}{suffix}{ext}"


def save_deflection_ridge_data(path, ridge_data, roi_theta, roi_pt,
                               collider, num_bunches, detector, field):
    """Persist deflection ridge information to JSON for later reuse."""
    if ridge_data is None:
        print("Warning: No deflection ridge available to save.")
        return

    payload = {
        'ridge_theta': np.asarray(ridge_data.get('ridge_theta', [])).tolist(),
        'ridge_pt': np.asarray(ridge_data.get('ridge_pt', [])).tolist(),
        'line_theta': None,
        'line_pt': None,
        'power_law': None,
        'roi_theta': list(roi_theta) if roi_theta else [None, None],
        'roi_pt': list(roi_pt) if roi_pt else [None, None],
        'metadata': {
            'collider': collider,
            'num_bunches': num_bunches,
            'detector': detector,
            'field_T': field,
            'saved_at_utc': datetime.utcnow().isoformat(timespec='seconds')
        }
    }

    if ridge_data.get('line_theta') is not None:
        payload['line_theta'] = np.asarray(ridge_data['line_theta']).tolist()
    if ridge_data.get('line_pt') is not None:
        payload['line_pt'] = np.asarray(ridge_data['line_pt']).tolist()
    if ridge_data.get('power_law') is not None:
        A, B = ridge_data['power_law']
        payload['power_law'] = [float(A), float(B)]

    try:
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"Deflection ridge saved to {path}")
    except OSError as exc:
        print(f"Warning: failed to save deflection ridge to {path}: {exc}")


def plot_pt_theta_scatter(pt_samples, theta_samples, base_output, collider, num_bunches, detector, field):
    """Save a scatter plot of pT versus theta for the produced particles."""
    if not base_output or not pt_samples or not theta_samples:
        return

    pt_array = np.asarray(pt_samples)
    theta_array = np.asarray(theta_samples)
    mask = np.isfinite(pt_array) & np.isfinite(theta_array)
    if not np.any(mask):
        return

    pt_array = pt_array[mask]
    theta_array = theta_array[mask]
    pt_array_mev = pt_array * 1e3

    if HAS_MPLHEP:
        hep.style.use("CMS")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(theta_array, pt_array_mev, s=5, alpha=0.2, linewidths=0)
    ax.set_xlabel(r'$\theta$ [rad]', fontsize=17)
    ax.set_ylabel(r'$p_{\mathrm{T}}$ [MeV/c]', fontsize=17)

    if num_bunches is None:
        bunch_count = 'N/A'
        bunch_text = 'bunches'
    else:
        bunch_text = 'bunch' if num_bunches == 1 else 'bunches'
        bunch_count = f"{num_bunches}"

    ax.set_title(
        f"{collider} ({bunch_count} {bunch_text})\nTransverse momentum vs polar angle",
        fontsize=18
    )
    ax.grid(True, alpha=0.3)

    output_path = _build_additional_output_path(base_output, '_pt_theta', forced_ext='.png')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"pT-theta scatter plot saved to {output_path}")

def plot_pt_theta_histogram(pt_samples, theta_samples, base_output, collider, num_bunches, detector, field,
                            reachability_boundary=None):
    """Save a log-log 2D histogram of pT versus theta for the produced particles."""
    if not base_output or not pt_samples or not theta_samples:
        return

    theta_array = np.asarray(theta_samples)
    pt_array = np.asarray(pt_samples)
    mask = np.isfinite(theta_array) & np.isfinite(pt_array) & (theta_array > 0) & (pt_array > 0)
    if not np.any(mask):
        return

    theta_array = theta_array[mask]
    pt_array = pt_array[mask]
    pt_array_mev = pt_array * 1e3

    if HAS_MPLHEP:
        hep.style.use("CMS")

    theta_min = max(theta_array.min(), 1e-6)
    theta_max = theta_array.max()
    pt_min = max(pt_array_mev.min(), 1e-6)
    pt_max = pt_array_mev.max()

    theta_bins = np.logspace(np.log10(theta_min), np.log10(theta_max), 80)
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 80)

    fig, ax = plt.subplots(figsize=(10, 7))
    hist = ax.hist2d(theta_array, pt_array_mev, bins=(theta_bins, pt_bins), norm=LogNorm())
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\theta$ [rad]', fontsize=17)
    ax.set_ylabel(r'$p_{\mathrm{T}}$ [MeV/c]', fontsize=17)

    if num_bunches is None:
        bunch_count = 'N/A'
        bunch_text = 'bunches'
    else:
        bunch_text = 'bunch' if num_bunches == 1 else 'bunches'
        bunch_count = f"{num_bunches}"

    ax.set_title(
        f"{collider} ({bunch_count} {bunch_text})\nTransverse momentum vs polar angle (2D histogram)",
        fontsize=18
    )
    ax.grid(True, which='both', alpha=0.3)

    cbar = fig.colorbar(hist[3], ax=ax)
    cbar.set_label('Entries/bin', fontsize=17)

    legend_entries = []
    legend_labels = []
    if reachability_boundary is not None:
        reach_theta, reach_pt = reachability_boundary
        reach_theta = np.asarray(reach_theta, dtype=float)
        reach_pt = np.asarray(reach_pt, dtype=float)
        valid = np.isfinite(reach_theta) & np.isfinite(reach_pt) & (reach_theta > 0) & (reach_pt > 0)
        reach_theta = reach_theta[valid]
        reach_pt = reach_pt[valid]
        if reach_theta.size > 0:
            x_top = ax.get_xlim()[1]
            if x_top > reach_theta[-1]:
                reach_theta = np.append(reach_theta, x_top)
                reach_pt = np.append(reach_pt, reach_pt[-1])
            reach_pt_mev = reach_pt * 1e3
            y_top = ax.get_ylim()[1]
            reach_line, = ax.plot(
                reach_theta,
                reach_pt_mev,
                color='black',
                linewidth=2.3,
                alpha=0.32,
                label='vertex reach boundary'
            )
            ax.fill_between(
                reach_theta,
                reach_pt_mev,
                np.full_like(reach_pt_mev, y_top),
                color='black',
                alpha=0.05,
                edgecolor='none'
            )
            legend_entries.append(reach_line)
            legend_labels.append('vertex reach boundary')

    if legend_entries:
        ax.legend(legend_entries, legend_labels, loc='upper left', fontsize=14, frameon=True, fancybox=True, shadow=True)

    output_path = _build_additional_output_path(base_output, '_pt_theta_hist2d', forced_ext='.pdf')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"pT-theta 2D histogram saved to {output_path}")



def _weighted_quantile(vals, weights, q):
    """Weighted quantile of 1D array vals with non-negative weights."""
    if vals.size == 0:
        return np.nan
    order = np.argsort(vals)
    v = vals[order]
    w = np.asarray(weights)[order]
    w = np.clip(w, 0, None)
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return np.nan
    cw /= cw[-1]
    return np.interp(q, cw, v)

def _compute_deflection_ridge(theta_samples, pt_samples,
                              n_theta_bins=120,
                              n_pt_bins=160,
                              min_col_counts=40,
                              max_jump_bins=4,
                              smooth_window=7,
                              fit_right_of_peak=True,
                              line_points=300,
                              roi_theta=(None, None),
                              roi_pt=(None, None)):
    """Estimate the beam-beam deflection ridge by tracking the dominant bin per θ column."""
    th = np.asarray(theta_samples)
    pt = np.asarray(pt_samples)
    m = np.isfinite(th) & np.isfinite(pt) & (th > 0) & (pt > 0)
    theta_min_roi, theta_max_roi = roi_theta if roi_theta is not None else (None, None)
    pt_min_roi, pt_max_roi = roi_pt if roi_pt is not None else (None, None)
    if theta_min_roi is not None:
        m &= th >= float(theta_min_roi)
    if theta_max_roi is not None:
        m &= th <= float(theta_max_roi)
    if pt_min_roi is not None:
        m &= pt >= float(pt_min_roi)
    if pt_max_roi is not None:
        m &= pt <= float(pt_max_roi)
    if not np.any(m):
        return None
    th = th[m]; pt = pt[m]
    lth = np.log10(th); lpt = np.log10(pt)

    # θ, pT binning in log10 space (respect ROI if provided)
    if theta_min_roi is not None:
        th_lo = np.log10(theta_min_roi)
    else:
        th_lo = np.percentile(lth, 0.5)
    if theta_max_roi is not None:
        th_hi = np.log10(theta_max_roi)
    else:
        th_hi = np.percentile(lth, 99.5)
    if pt_min_roi is not None:
        pt_lo = np.log10(pt_min_roi)
        pt_lo_log = pt_lo
    else:
        pt_lo = np.percentile(lpt, 0.5)
        pt_lo_log = None
    if pt_max_roi is not None:
        pt_hi = np.log10(pt_max_roi)
        pt_hi_log = pt_hi
    else:
        pt_hi = np.percentile(lpt, 99.5)
        pt_hi_log = None
    if th_hi <= th_lo or pt_hi <= pt_lo:
        return None
    th_edges = np.linspace(th_lo, th_hi, n_theta_bins + 1)
    pt_edges = np.linspace(pt_lo, pt_hi, n_pt_bins + 1)
    th_cent = 0.5 * (th_edges[:-1] + th_edges[1:])
    pt_cent = 0.5 * (pt_edges[:-1] + pt_edges[1:])

    H, _, _ = np.histogram2d(lth, lpt, bins=(th_edges, pt_edges))  # shape: (n_theta_bins, n_pt_bins)

    column_data = []
    common_mask = np.ones_like(pt_cent, dtype=bool)
    if pt_lo_log is not None:
        common_mask &= pt_cent >= pt_lo_log
    if pt_hi_log is not None:
        common_mask &= pt_cent <= pt_hi_log

    for i in range(n_theta_bins):
        col = H[i, :]
        if not np.any(common_mask):
            continue
        ptc = pt_cent[common_mask]
        counts = col[common_mask]
        if counts.sum() < min_col_counts:
            continue
        column_data.append((i, ptc, counts))

    if not column_data:
        return None

    target_log_theta = np.log10(0.1)
    seed_entry = min(column_data, key=lambda entry: abs(th_cent[entry[0]] - target_log_theta))
    seed_idx = seed_entry[0]
    seed_ptc = seed_entry[1]
    seed_counts = seed_entry[2]

    sorted_bins = np.argsort(seed_counts)[::-1]
    seed_bin = next((b for b in sorted_bins if seed_counts[b] > 0), None)
    if seed_bin is None:
        return None

    ridge_bins = {seed_idx: seed_bin}

    sorted_indices = [entry[0] for entry in column_data]
    index_map = {entry[0]: entry for entry in column_data}
    sorted_indices.sort()
    seed_pos = sorted_indices.index(seed_idx)

    prev_bin = seed_bin
    for idx in sorted_indices[seed_pos + 1:]:
        _, ptc, counts = index_map[idx]
        candidates = np.argsort(counts)[::-1]
        chosen = next((c for c in candidates if counts[c] > 0 and abs(c - prev_bin) <= max_jump_bins), None)
        if chosen is None:
            chosen = next((c for c in candidates if counts[c] > 0), None)
        if chosen is None:
            continue
        ridge_bins[idx] = chosen
        prev_bin = chosen

    prev_bin = seed_bin
    for idx in reversed(sorted_indices[:seed_pos]):
        _, ptc, counts = index_map[idx]
        candidates = np.argsort(counts)[::-1]
        chosen = next((c for c in candidates if counts[c] > 0 and abs(c - prev_bin) <= max_jump_bins), None)
        if chosen is None:
            chosen = next((c for c in candidates if counts[c] > 0), None)
        if chosen is None:
            continue
        ridge_bins[idx] = chosen
        prev_bin = chosen

    if len(ridge_bins) < 5:
        return None

    ridge_th_log = np.array([th_cent[idx] for idx in sorted(ridge_bins.keys())])
    ridge_pt_log = np.array([index_map[idx][1][ridge_bins[idx]] for idx in sorted(ridge_bins.keys())])

    # smooth (simple moving average in log space)
    if smooth_window >= 3 and smooth_window % 2 == 1 and len(ridge_pt_log) >= smooth_window:
        k = np.ones(smooth_window, float) / smooth_window
        sm = np.convolve(ridge_pt_log, k, mode='same')
        half = smooth_window // 2
        sm[:half] = ridge_pt_log[:half]
        sm[-half:] = ridge_pt_log[-half:]
        ridge_pt_log_s = sm
    else:
        ridge_pt_log_s = ridge_pt_log

    # choose fit region: right of peak (monotone decreasing branch)
    A_B = None
    line_th_log = None
    line_pt_log = None
    if fit_right_of_peak and len(ridge_pt_log_s) >= 8:
        ipk = int(np.nanargmax(ridge_pt_log_s))
        fit_idx = np.arange(ipk + 2, len(ridge_pt_log_s) - 1)  # skip the top & far end
        if fit_idx.size >= 5:
            x = ridge_th_log[fit_idx]
            y = ridge_pt_log_s[fit_idx]
            # robust-ish: clip extreme outliers (rare top spikes)
            med = np.median(y)
            iqr = np.percentile(y, 75) - np.percentile(y, 25)
            keep = (y > med - 2.0*iqr) & (y < med + 2.0*iqr)
            x = x[keep]; y = y[keep]
            if x.size >= 5:
                B, a = np.polyfit(x, y, 1)   # y = a + B x
                A = 10**a
                A_B = (A, B)
                line_th_log = np.linspace(x.min(), x.max(), line_points)
                line_pt_log = a + B * line_th_log

    return {
        'ridge_theta': 10**ridge_th_log,
        'ridge_pt'   : 10**ridge_pt_log_s,
        'line_theta' : None if line_th_log is None else 10**line_th_log,
        'line_pt'    : None if line_pt_log  is None else 10**line_pt_log,
        'power_law'  : A_B
    }


def plot_pt_theta_histogram_with_deflection(theta_samples, pt_samples, base_output,
                                            collider, num_bunches, detector, field,
                                            roi_theta=(None, None), roi_pt=(None, None),
                                            ridge_data=None,
                                            reachability_boundary=None):
    """Draw the 2D histogram with the beam-beam deflection ridge highlighted."""
    if not base_output or not pt_samples or not theta_samples:
        return

    if roi_theta is None:
        roi_theta = (None, None)
    if roi_pt is None:
        roi_pt = (None, None)

    theta_array = np.asarray(theta_samples)
    pt_array = np.asarray(pt_samples)
    full_mask = np.isfinite(theta_array) & np.isfinite(pt_array) & (theta_array > 0) & (pt_array > 0)
    if not np.any(full_mask):
        return

    theta_full = theta_array[full_mask]
    pt_full = pt_array[full_mask]
    pt_full_mev = pt_full * 1e3

    if theta_full.size == 0 or pt_full.size == 0:
        return

    if HAS_MPLHEP:
        hep.style.use("CMS")

    theta_min = theta_full.min()
    theta_max = theta_full.max()
    pt_min = pt_full_mev.min()
    pt_max = pt_full_mev.max()

    if theta_min <= 0 or pt_min <= 0 or theta_max <= theta_min or pt_max <= pt_min:
        return

    theta_bins = np.logspace(np.log10(theta_min), np.log10(theta_max), 80)
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 80)

    fig, ax = plt.subplots(figsize=(10, 7))
    hist = ax.hist2d(theta_full, pt_full_mev, bins=(theta_bins, pt_bins), norm=LogNorm())
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\theta$ [rad]', fontsize=17)
    ax.set_ylabel(r'$p_{\mathrm{T}}$ [MeV/c]', fontsize=17)

    if num_bunches is None:
        bunch_count = 'N/A'
        bunch_text = 'bunches'
    else:
        bunch_text = 'bunch' if num_bunches == 1 else 'bunches'
        bunch_count = f"{num_bunches}"

    ax.set_title(
        f"{collider} ({bunch_count} {bunch_text})",
        fontsize=18
    )
    ax.grid(True, which='both', alpha=0.3)

    cbar = fig.colorbar(hist[3], ax=ax)
    cbar.set_label('Entries/bin', fontsize=17)

    legend_entries = []
    if ridge_data is not None:
        ridge_theta = np.asarray(ridge_data.get('ridge_theta', []))
        ridge_pt = np.asarray(ridge_data.get('ridge_pt', []))
        ridge_pt_mev = ridge_pt * 1e3
        valid_ridge = (
            np.isfinite(ridge_theta)
            & np.isfinite(ridge_pt)
            & (ridge_theta > 0)
            & (ridge_pt > 0)
        )
        ridge_theta = ridge_theta[valid_ridge]
        ridge_pt_mev = ridge_pt_mev[valid_ridge]

        if ridge_theta.size > 0:
            ridge_line, = ax.plot(
                ridge_theta,
                ridge_pt_mev,
                color='red',
                linewidth=2,
                label='Deflection ridge'
            )
            legend_entries.append(ridge_line)

        line_theta = ridge_data.get('line_theta')
        line_pt = ridge_data.get('line_pt')
        power_law = ridge_data.get('power_law')
        if line_theta is not None and line_pt is not None:
            line_theta = np.asarray(line_theta)
            line_pt = np.asarray(line_pt)
            line_pt_mev = line_pt * 1e3
            valid_line = (
                np.isfinite(line_theta)
                & np.isfinite(line_pt)
                & (line_theta > 0)
                & (line_pt > 0)
            )
            line_theta = line_theta[valid_line]
            line_pt_mev = line_pt_mev[valid_line]
            if line_theta.size > 0:
                power_fit_color = 'tab:brown'
                power_line, = ax.plot(
                    line_theta,
                    line_pt_mev,
                    color=power_fit_color,
                    linestyle='--',
                    linewidth=2,
                    label=r"Power-law fit"
                )
                legend_entries.append(power_line)
                if power_law is not None:
                    A, B = power_law
                    A_mev = A * 1e3
                    text = fr"$p_{{\mathrm{{T}}}}\,[\mathrm{{MeV}}/c] = {A_mev:.2g}\,(\theta\,[\mathrm{{rad}}])^{{{B:.2f}}}$"
                    ax.text(
                        0.02,
                        0.70,
                        text,
                        transform=ax.transAxes,
                        fontsize=15,
                        fontweight='bold',
                        color=power_fit_color,
                        bbox=dict(facecolor='white', alpha=0.05, edgecolor='none')
                    )
    else:
        print("Warning: Unable to determine deflection ridge; saving histogram without overlay.")

    if reachability_boundary is not None:
        reach_theta, reach_pt = reachability_boundary
        reach_theta = np.asarray(reach_theta, dtype=float)
        reach_pt = np.asarray(reach_pt, dtype=float)
        valid = np.isfinite(reach_theta) & np.isfinite(reach_pt) & (reach_theta > 0) & (reach_pt > 0)
        reach_theta = reach_theta[valid]
        reach_pt = reach_pt[valid]
        if reach_theta.size > 0:
            x_top = ax.get_xlim()[1]
            if x_top > reach_theta[-1]:
                reach_theta = np.append(reach_theta, x_top)
                reach_pt = np.append(reach_pt, reach_pt[-1])
            reach_pt_mev = reach_pt * 1e3
            y_top = ax.get_ylim()[1]
            reach_line, = ax.plot(
                reach_theta,
                reach_pt_mev,
                color='black',
                linewidth=2.3,
                alpha=0.32,
                label='vertex reach boundary'
            )
            ax.fill_between(
                reach_theta,
                reach_pt_mev,
                np.full_like(reach_pt_mev, y_top),
                color='black',
                alpha=0.05,
                edgecolor='none'
            )
            legend_entries.append(reach_line)

    legend_obj = None
    if legend_entries:
        legend_obj = ax.legend(loc='upper left',fontsize=16)

    output_path = _build_additional_output_path(base_output, '_pt_theta_hist2d_deflection', forced_ext='.pdf')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"pT-theta deflection histogram saved to {output_path}")


def main():
    """Main function to process files and generate plots."""
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_args()
    
    # Find .dat files in input directory
    dat_files = [f for f in os.listdir(args.indir) if f.startswith('pairs_') and f.endswith('.dat')]
    #dat_files = [f for f in os.listdir(args.indir) if f.startswith('testC3_250_pairs_particles_0.62_emittx_0.40_emitty_0.02_betax_12.0_betay_0.12_sigmaz_100.0_offsety_0.0_seed_') and f.endswith('.dat')]

    if not dat_files:
        print(f"No .dat files found in {args.indir}")
        return
    
    # Limit number of files if needed
    dat_files = [os.path.join(args.indir, f) for f in sorted(dat_files)[:args.max_files]]
    print(f"Will process {len(dat_files)} .dat files")
    
    # Setup histogram dimensions
    nz = args.nz
    nr = args.nr
    z_min = args.zmin
    z_max = args.zmax
    r_min = args.rmin
    r_max = args.rmax
    
    # Decide if we need to keep trajectories
    keep_trajectories = args.plot_trajectories > 0
    # Track what to display for bunch count in the title
    display_bunches = args.num_bunches

    roi_theta = (args.roi_theta_min, args.roi_theta_max)
    roi_pt = (args.roi_pt_min, args.roi_pt_max)

    ridge_output_path = args.save_deflection_ridge
    if ridge_output_path is not None:
        ridge_output_path = ridge_output_path.strip()
        if ridge_output_path == '':
            ridge_output_path = None
    if ridge_output_path is None and args.out:
        ridge_output_path = _build_additional_output_path(args.out, '_deflection_ridge', forced_ext='.json')
    
    # Check for previously saved results to reuse
    cache_file = None
    if args.cache_trajectory_data:
        cache_file = os.path.join(args.indir, f"cached_results_{args.field}T_{nz}x{nr}_{z_min}to{z_max}_{args.max_files}_bunches.pkl")
        if os.path.exists(cache_file):
            print(f"Loading cached results from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    combined_hist = cached_data['combined_hist']
                    all_trajectories = cached_data['all_trajectories']
                    total_pass = cached_data['total_pass']
                    total_particles = cached_data['total_particles']
                    combined_hist.setdefault('pt_values', [])
                    combined_hist.setdefault('theta_values', [])
                    # Optionally normalize histograms to per-bunch
                    normalization_files = cached_data.get('num_files', None)
                    if args.normalize_per_bunch:
                        if normalization_files is not None and normalization_files > 0:
                            combined_hist['r_vs_z'] = combined_hist['r_vs_z'] / normalization_files
                            combined_hist['x_vs_z'] = combined_hist['x_vs_z'] / normalization_files
                            combined_hist['y_vs_z'] = combined_hist['y_vs_z'] / normalization_files
                            display_bunches = 1
                        elif args.num_bunches and args.num_bunches > 0:
                            # Fallback: use user-provided number of bunches
                            combined_hist['r_vs_z'] = combined_hist['r_vs_z'] / args.num_bunches
                            combined_hist['x_vs_z'] = combined_hist['x_vs_z'] / args.num_bunches
                            combined_hist['y_vs_z'] = combined_hist['y_vs_z'] / args.num_bunches
                            display_bunches = 1
                    else:
                        # Not normalizing: display the number of aggregated bunches, if known
                        if normalization_files is not None and normalization_files > 0:
                            display_bunches = normalization_files
                        else:
                            display_bunches = args.num_bunches
                    
                    print(f"Loaded cached data for {total_particles} particles")
                    print(f"Particles passing selection: {total_pass} ({total_pass/total_particles*100:.2f}%)")
                    
                    # Skip to envelope calculation
                    if args.coord == 'r':
                        histogram = combined_hist['r_vs_z']
                    elif args.coord == 'x':
                        histogram = combined_hist['x_vs_z']
                    else:  # args.coord == 'y'
                        histogram = combined_hist['y_vs_z']
                    
                    # Calculate envelopes
                    envelopes = calculate_envelopes(histogram, args.percentiles, z_min, z_max, args.smooth_envelopes)
                    
                    # Save envelopes if requested
                    if args.save_envelopes:
                        with open(args.save_envelopes, 'wb') as f:
                            pickle.dump(envelopes, f)
                        print(f"Envelopes saved to {args.save_envelopes}")

                    theta_samples_plot = combined_hist.get('theta_values', [])
                    pt_samples_plot = combined_hist.get('pt_values', [])
                    ridge_data_cached = None
                    if theta_samples_plot and pt_samples_plot:
                        ridge_data_cached = _compute_deflection_ridge(
                            theta_samples_plot,
                            pt_samples_plot,
                            roi_theta=roi_theta,
                            roi_pt=roi_pt
                        )
                        if ridge_data_cached is not None and ridge_output_path:
                            save_deflection_ridge_data(
                                ridge_output_path,
                                ridge_data_cached,
                                roi_theta,
                                roi_pt,
                                args.collider,
                                display_bunches,
                                args.detector,
                                args.field
                            )
                    else:
                        ridge_data_cached = None

                    reachability_boundary = None
                    if args.show_reachability_boundary:
                        pt_array_hint = np.asarray(pt_samples_plot, dtype=float)
                        valid_hint = np.isfinite(pt_array_hint) & (pt_array_hint > 0)
                        pt_upper_hint = float(pt_array_hint[valid_hint].max()) if np.any(valid_hint) else None
                        boundary_theta, boundary_pt = _sample_reachability_boundary(args, pt_upper_hint)
                        if boundary_theta is not None and boundary_pt is not None:
                            reachability_boundary = (boundary_theta, boundary_pt)

                    # Plot results
                    plot_envelopes(
                        envelopes,
                        histogram=histogram,
                        trajectories=all_trajectories[:args.plot_trajectories] if keep_trajectories else None,
                        coord=args.coord,
                        z_min=z_min,
                        z_max=z_max,
                        draw_2d_histo=args.draw_2d_histo,
                        output_file=args.out,
                        percentiles=args.percentiles,
                        collider=args.collider,
                        num_bunches=display_bunches,
                        detector=args.detector,
                        field=args.field,
                        show_detector=args.show_detector,
                        show_colorbar=args.show_colorbar,
                        colorbar_min=args.colorbar_min,
                        colorbar_max=args.colorbar_max,
                        density_units=args.density_units,
                        cmap_name=args.cmap,
                        r_min=r_min,
                        r_max=r_max
                    )

                    plot_pt_theta_scatter(
                        pt_samples_plot,
                        theta_samples_plot,
                        args.out,
                        collider=args.collider,
                        num_bunches=display_bunches,
                        detector=args.detector,
                        field=args.field
                    )

                    plot_pt_theta_histogram(
                        pt_samples_plot,
                        theta_samples_plot,
                        args.out,
                        collider=args.collider,
                        num_bunches=display_bunches,
                        detector=args.detector,
                        field=args.field,
                        reachability_boundary=reachability_boundary
                    )

                    plot_pt_theta_histogram_with_deflection(
                        theta_samples_plot,
                        pt_samples_plot,
                        args.out,
                        collider=args.collider,
                        num_bunches=display_bunches,
                        detector=args.detector,
                        field=args.field,
                        roi_theta=roi_theta,
                        roi_pt=roi_pt,
                        ridge_data=ridge_data_cached,
                        reachability_boundary=reachability_boundary
                    )

                    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
                    return
            except Exception as e:
                print(f"Error loading cached data: {e}, will recompute")
    
    # Process files
    if args.parallel:
        print("Using parallel processing")
        combined_hist, all_trajectories, total_pass, total_particles = process_files_parallel(
            dat_files, args.field, nz, nr, z_min, z_max, r_min, r_max, args.pt_cut, keep_trajectories,
            num_threads=args.parallel_threads
        )
    else:
        print("Using sequential processing")
        combined_hist = None
        all_trajectories = []
        total_pass = 0
        total_particles = 0
        
        for dat_file in dat_files:
            hist, trajectories, pass_count, total_count = process_dat_file(
                dat_file, args.field, nz, nr, z_min, z_max, r_min, r_max, args.pt_cut, keep_trajectories
            )
            
            # Initialize or update combined histograms
            if combined_hist is None:
                combined_hist = hist
                combined_hist.setdefault('pt_values', [])
                combined_hist.setdefault('theta_values', [])
            else:
                combined_hist.setdefault('pt_values', [])
                combined_hist.setdefault('theta_values', [])
                combined_hist['r_vs_z'] += hist['r_vs_z']
                combined_hist['x_vs_z'] += hist['x_vs_z']
                combined_hist['y_vs_z'] += hist['y_vs_z']
                combined_hist['r_plus_62'].extend(hist['r_plus_62'])
                combined_hist['r_minus_62'].extend(hist['r_minus_62'])
                combined_hist['r_max'].extend(hist['r_max'])
                combined_hist['pt_values'].extend(hist.get('pt_values', []))
                combined_hist['theta_values'].extend(hist.get('theta_values', []))
            
            all_trajectories.extend(trajectories)
            total_pass += pass_count
            total_particles += total_count
            
            print(f"Processed {dat_file}: {pass_count}/{total_count} particles pass selection ({pass_count/total_count*100:.2f}%)")
    
    print(f"\nTotal particles: {total_particles}")
    print(f"Particles passing selection: {total_pass} ({total_pass/total_particles*100:.2f}%)")
    
    # Save results to cache if requested
    if args.cache_trajectory_data and cache_file:
        print(f"Saving results to cache file {cache_file}")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'combined_hist': combined_hist,
                    'all_trajectories': all_trajectories,
                    'total_pass': total_pass,
                    'total_particles': total_particles,
                    'num_files': len(dat_files)
                }, f)
            print("Cache saved successfully")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    # Normalize histograms (fresh run) if requested
    if len(dat_files) > 0:
        if args.normalize_per_bunch:
            combined_hist['r_vs_z'] /= len(dat_files)
            combined_hist['x_vs_z'] /= len(dat_files)
            combined_hist['y_vs_z'] /= len(dat_files)
            # We normalized to per-bunch; reflect that in the title
            display_bunches = 1
        else:
            # No normalization: show the aggregated bunch count (files processed or provided label)
            display_bunches = len(dat_files) if args.num_bunches is None else args.num_bunches
    
    # Select histogram based on coordinate
    if args.coord == 'r':
        histogram = combined_hist['r_vs_z']
    elif args.coord == 'x':
        histogram = combined_hist['x_vs_z']
    else:  # args.coord == 'y'
        histogram = combined_hist['y_vs_z']
    
    # Calculate envelopes
    envelopes = calculate_envelopes(histogram, args.percentiles, z_min, z_max, args.smooth_envelopes)
    
    # Save envelopes if requested
    if args.save_envelopes:
        with open(args.save_envelopes, 'wb') as f:
            pickle.dump(envelopes, f)
        print(f"Envelopes saved to {args.save_envelopes}")

    theta_samples_plot = combined_hist.get('theta_values', [])
    pt_samples_plot = combined_hist.get('pt_values', [])
    ridge_data_final = None
    if theta_samples_plot and pt_samples_plot:
        ridge_data_final = _compute_deflection_ridge(
            theta_samples_plot,
            pt_samples_plot,
            roi_theta=roi_theta,
            roi_pt=roi_pt
        )
        if ridge_data_final is not None and ridge_output_path:
            save_deflection_ridge_data(
                ridge_output_path,
                ridge_data_final,
                roi_theta,
                roi_pt,
                args.collider,
                display_bunches,
                args.detector,
                args.field
            )

    reachability_boundary = None
    if args.show_reachability_boundary:
        pt_array_hint = np.asarray(pt_samples_plot, dtype=float)
        valid_hint = np.isfinite(pt_array_hint) & (pt_array_hint > 0)
        pt_upper_hint = float(pt_array_hint[valid_hint].max()) if np.any(valid_hint) else None
        boundary_theta, boundary_pt = _sample_reachability_boundary(args, pt_upper_hint)
        if boundary_theta is not None and boundary_pt is not None:
            reachability_boundary = (boundary_theta, boundary_pt)

    # Plot results
    plot_envelopes(
        envelopes,
        histogram=histogram,
        trajectories=all_trajectories[:args.plot_trajectories] if keep_trajectories else None,
        coord=args.coord,
        z_min=z_min,
        z_max=z_max,
        draw_2d_histo=args.draw_2d_histo,
        output_file=args.out,
        percentiles=args.percentiles,
        collider=args.collider,
        num_bunches=display_bunches,
        detector=args.detector,
        field=args.field,
        show_detector=args.show_detector,
        r_min=r_min,
        r_max=r_max,
        density_units=args.density_units,
        cmap_name=args.cmap
    )

    plot_pt_theta_scatter(
        pt_samples_plot,
        theta_samples_plot,
        args.out,
        collider=args.collider,
        num_bunches=display_bunches,
        detector=args.detector,
        field=args.field
    )

    plot_pt_theta_histogram(
        pt_samples_plot,
        theta_samples_plot,
        args.out,
        collider=args.collider,
        num_bunches=display_bunches,
        detector=args.detector,
        field=args.field,
        reachability_boundary=reachability_boundary
    )

    plot_pt_theta_histogram_with_deflection(
        theta_samples_plot,
        pt_samples_plot,
        args.out,
        collider=args.collider,
        num_bunches=display_bunches,
        detector=args.detector,
        field=args.field,
        roi_theta=roi_theta,
        roi_pt=roi_pt,
        ridge_data=ridge_data_final,
        reachability_boundary=reachability_boundary
    )

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

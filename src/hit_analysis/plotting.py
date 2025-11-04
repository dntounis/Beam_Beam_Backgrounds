import re

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import traceback

import mplhep as hep 

from src.detector_config import get_detector_configs, get_xmls
from src.geometry_parsing.geometry_info import get_geometry_info

from src.hit_analysis.occupancy import analyze_detector_hits
from src.geometry_parsing.k4geo_parsers import parse_detector_constants 
from src.utils.histogram_utils import (
    compute_rphi_area_map,
    compute_rz_area_map,
    extract_layer_geometry,
    filter_hits_to_geometry,
)


def _configure_polar_axis(ax, r_values, theta_step=45, r_label_angle=135):
    """Tidy polar layout so angular labels and radial ticks avoid overlap."""

    r_array = np.asarray(r_values)
    r_array = r_array[np.isfinite(r_array)]
    if r_array.size == 0:
        r_max = 1.0
    else:
        r_max = float(np.max(r_array))
        if r_max <= 0:
            r_max = 1.0

    margin = 0.05 * r_max
    ax.set_ylim(0, r_max + margin)

    theta = np.arange(0, 360, theta_step)
    theta_rad = np.deg2rad(theta)
    ax.set_xticks(theta_rad)
    ax.set_xticklabels([])
    # Draw custom angular labels oriented tangent to the circle
    for angle, angle_rad in zip(theta, theta_rad):
        deg = int(angle) % 360
        rotation = np.rad2deg(angle_rad) + 90
        if deg in (0, 180):
            rotation = 0
        elif 45 <= deg <= 135:
            rotation -= 180
        ax.text(angle_rad, r_max + margin * 3, f"{deg}°",
                rotation=rotation, rotation_mode='anchor',
                ha='center', va='center', fontsize=12)

    ax.tick_params(axis='x', pad=12)
    ax.tick_params(axis='y', pad=6)
    ax.set_rlabel_position(r_label_angle)

    r_ticks = np.linspace(r_max / 4.0, r_max, 3)
    if r_max >= 50:
        step = 5
    elif r_max >= 20:
        step = 2
    else:
        step = 1
    r_ticks = np.round(r_ticks / step) * step
    r_ticks = np.unique(r_ticks)
    r_ticks = r_ticks[r_ticks > 0]
    if r_ticks.size:
        tick_labels = [f"{tick:.0f}" if tick >= 5 else f"{tick:.1f}" for tick in r_ticks]
        ax.set_rgrids(r_ticks, labels=['' for _ in r_ticks], angle=r_label_angle)
        angle_rad = np.deg2rad(r_label_angle)
        for tick, label in zip(r_ticks, tick_labels):
            ax.text(angle_rad, tick + 0.015 * r_max, label,
                    ha='center', va='center', fontsize=12,
                    rotation=r_label_angle - 90,
                    rotation_mode='anchor',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))
        # Lightly emphasize the guide line used for labels
        ax.plot([angle_rad, angle_rad], [0, r_max + margin],
                color='k', alpha=0.6, linewidth=1.0, linestyle='--', zorder=2)

    ax.yaxis.set_label_coords(-0.2, 0.80)
    ax.yaxis.label.set_rotation(90)
    ax.yaxis.label.set_verticalalignment('center')
    ax.yaxis.label.set_horizontalalignment('center')


def _configure_log_yticks(ax):
    """Ensure consistent log-scale major/minor ticks across plots."""
    try:
        from matplotlib.ticker import LogLocator, NullFormatter, LogFormatterSciNotation

        major_locator = LogLocator(base=10.0, numticks=1000)
        minor_locator = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=1000)

        ax.yaxis.set_major_locator(major_locator)
        ax.yaxis.set_minor_locator(minor_locator)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        ax.tick_params(axis='y', which='minor', length=4, width=0.8)
        ax.grid(True, which='both', axis='y', alpha=0.3)
    except Exception:
        pass


_DISPLAY_NAME_OVERRIDES = {
    'SiTrackerForward': 'SiVertexForward',
}


def _get_detector_display_name(detector_name: str) -> str:
    """
    Return the user-facing label for a detector, applying overrides when needed.
    """
    return _DISPLAY_NAME_OVERRIDES.get(detector_name, detector_name)

# import matplotlib as mpl
# mpl.rc('text', usetex=True)
# mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


def plot_hit_distribution(stats, output_file=None):
    """
    Create visualization of hit distribution
    
    Parameters:
    -----------
    stats : dict
        Statistics from analyze_vertex_detector
    output_file : str, optional
        If provided, save plot to this file
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Hit distribution histogram
    hits, counts = zip(*sorted(stats['hit_distribution'].items()))
    ax1.bar(hits, counts)
    ax1.set_xlabel('Hits per cell',fontsize=18)
    ax1.set_ylabel('Number of cells',fontsize=18)
    #ax1.set_title('Hit Distribution',fontsize=20)
    ax1.set_yscale('log')
    
    # Per-layer statistics
    layers = sorted(stats['per_layer'].keys())
    #occupancies = [stats['per_layer'][layer]['cells_hit'] for layer in layers]
    occupancies = [stats['per_layer'][layer]['pixels_hit'] for layer in layers]

    ax2.bar(layers, occupancies)
    ax2.set_xlabel('Layer',fontsize=18)
    ax2.set_ylabel('Number of cells hit',fontsize=18)
    #ax2.set_title('Cells Hit per Layer',fontsize=20)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
    plt.show()



def plot_occupancy_analysis(stats, geometry_info, output_prefix=None, time_cut=-1, nlayer_batch=1,
                            occupancy_scale=1.0, geometry_filter_tolerance=1.0):
    """
    Create detailed visualizations of the occupancy analysis
    
    Parameters:
    -----------
    stats : dict
        Statistics from analyze_vertex_detector
    geometry_info : dict
        Geometry information
    output_prefix : str, optional
        If provided, save plots with this prefix
    time_cut : float, optional
        Cut on hit time in ns (-1 for no cut)
    nlayer_batch : int, optional
    geometry_filter_tolerance : float, optional
        Maximum (mm) a hit may lie outside the parsed geometry before being
        discarded from R-phi/R-z/timing visualizations.
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))



    plt.style.use(hep.style.CMS)
    # plt.rcParams.update({
    #     'font.family': 'serif',
    #     'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'serif'],
    #     'mathtext.fontset': 'dejavuserif',
    # })


    # Retrieve assumed time cut for occupancy calculation
    time_cut = stats['time_cut']
    print(f"Time cut: {time_cut}")

    detector_label = _get_detector_display_name(geometry_info["detector_name"])
    title_text = f'SiD_o2_v04 - {detector_label}'
    if time_cut > 0:
        title_text += f' (t<{time_cut} ns)'

    # Add the C^3 info and use rich text formatting for "Preliminary"
    #title_text += r'                   SiD_o2_v04@C$^{3}$-550 (266 bunches) - $\boldsymbol{\it{Preliminary}}$'
    #title_text += r'                   C$^{3}$ 550 - s.u. (150 bunches) - $\boldsymbol{\it{Preliminary}}$'
    #title_text += r'                   C$^{3}$ 250 - s.u. (266 bunches) - $\boldsymbol{\it{Preliminary}}$'
    title_text += r'                   C$^{3}$ 250 - Baseline (133 bunches) - $\boldsymbol{\it{Preliminary}}$'

    fig.suptitle(title_text, fontsize=24)

    gs = plt.GridSpec(2, 2)
    
    # 1. Occupancy vs threshold for each layer
    ax1 = fig.add_subplot(gs[0, 0])
    scale = occupancy_scale if occupancy_scale is not None else 1.0
    series_data = []
    thresholds = sorted(stats['threshold_stats'].keys())

    # Get all available layers and sort them
    all_layers = sorted(geometry_info['layers'].keys())
    
    # Create layer batches
    layer_batches = []
    for i in range(0, len(all_layers), nlayer_batch):
        batch = all_layers[i:i+nlayer_batch]
        if batch:  # Skip empty batches
            layer_batches.append(batch)

    # Plot occupancy with error bars for each batch
    for i, batch in enumerate(layer_batches):
        # Label for the batch
        if len(batch) == 1:
            batch_label = f'Layer {batch[0]}'
        else:
            batch_label = f'Layers {batch[0]}-{batch[-1]}'
        
        # Calculate average occupancy and errors for each threshold
        avg_occupancies = []
        occupancy_errors = []
        
        for t in thresholds:
            # Get all relevant stats for layers in this batch
            batch_stats = []
            
            for layer in batch:
                if layer in stats['threshold_stats'][t]['per_layer']:
                    layer_stats = stats['threshold_stats'][t]['per_layer'][layer]
                    
                    # Extract key statistics
                    cells_hit = layer_stats.get('cells_above_threshold', layer_stats.get('cells_hit', 0))
                    total_cells = geometry_info['layers'][layer].get('total_cells', 1)
                    occupancy = cells_hit / total_cells if total_cells > 0 else 0
                    
                    # Store stats for this layer
                    batch_stats.append({
                        'cells_hit': cells_hit,
                        'total_cells': total_cells,
                        'occupancy': occupancy
                    })
            
            # Calculate average occupancy across batch
            if batch_stats:
                # Method 1: Simple average of occupancies
                avg_occ = sum(s['occupancy'] for s in batch_stats) / len(batch_stats)
                
                # Calculate statistical error
                # For binomial statistics, error = sqrt(p*(1-p)/N) where p is occupancy and N is total cells
                # For small occupancies, approximate as sqrt(p/N)
                total_cells_hit = sum(s['cells_hit'] for s in batch_stats)
                total_cells_all = sum(s['total_cells'] for s in batch_stats)
                
                # Use Poisson approximation for low occupancy
                if avg_occ > 0:
                    # Standard error: sqrt(p*(1-p)/N)
                    # For low occupancy, p*(1-p) ≈ p
                    error = np.sqrt(avg_occ * (1 - avg_occ) / total_cells_all)
                    
                    # For very low occupancy where standard error might be unreliable:
                    if total_cells_hit < 10:  
                        # Alternative: use sqrt(k)/N where k is cell count (Poisson error)
                        error = np.sqrt(total_cells_hit) / total_cells_all
                else:
                    # Handle zero occupancy case (upper limit)
                    error = 1.0 / total_cells_all if total_cells_all > 0 else 0
            else:
                avg_occ = 0
                error = 0
                
            avg_occupancies.append(avg_occ)
            occupancy_errors.append(error)
        
        # Plot with error bars
        scaled_avg = [val * scale for val in avg_occupancies]
        scaled_err = [err * scale for err in occupancy_errors]

        ax1.errorbar(
            thresholds,
            scaled_avg,
            yerr=scaled_err,
            fmt='o-',
            label=batch_label,
            capsize=3
        )
        series_data.append((batch_label, scaled_avg, scaled_err))
    

 
    ax1.set_xlabel('Buffer depth',fontsize=18)
    ax1.set_ylabel('Layer occupancy',fontsize=18)
    ax1.set_yscale('log')
    _configure_log_yticks(ax1)
    #ax1.set_title('Occupancy vs Hit Threshold by Layer',fontsize=20)
    ax1.set_xticks(thresholds)  # Set ticks to the actual threshold values
    ax1.set_xticklabels([str(int(t)) for t in thresholds])  # Format as integers
    ax1.grid(True)
    ax1.legend()
    
    layer_metadata = extract_layer_geometry(geometry_info)

    # Debug: inspect surviving hits before geometry masking
    try:
        debug_count = len(r_vals)
        geom_mask = filter_hits_to_geometry(r_vals, z_vals, layer_metadata,
                                            tolerance=geometry_filter_tolerance)
        debug_filtered = int(np.count_nonzero(geom_mask)) if hasattr(np, 'count_nonzero') else int(sum(geom_mask))
        print("JIM DEBUG [geom-filter]", detector_label,
              "hits before mask:", debug_count,
              "after mask:", debug_filtered)
    except Exception:
        pass

    # 2. R-Phi hit distribution
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    # Convert awkward arrays to numpy arrays
    phi_vals = ak.to_numpy(stats['positions']['phi'])
    r_vals = ak.to_numpy(stats['positions']['r'])
    z_vals = ak.to_numpy(stats['positions']['z'])

    geom_mask = filter_hits_to_geometry(r_vals, z_vals, layer_metadata,
                                        tolerance=geometry_filter_tolerance)
    phi_filtered = phi_vals[geom_mask] if geom_mask.size else phi_vals
    r_filtered = r_vals[geom_mask] if geom_mask.size else r_vals

    # Create the edges first
    hist, xedges, yedges = np.histogram2d(phi_filtered, r_filtered, bins=[51, 21])

    area_map = compute_rphi_area_map(layer_metadata, xedges, yedges)
    with np.errstate(invalid='ignore', divide='ignore'):
        hist_normalized = np.divide(hist, area_map, where=area_map > 0)
    masked = np.ma.masked_where(hist_normalized <= 0, hist_normalized)
    cmap = plt.cm.get_cmap('viridis').copy()
    cmap.set_bad(color='white')
    pcm = ax2.pcolormesh(xedges[:-1], yedges[:-1], masked.T,
                         shading='auto', cmap=cmap,
                         edgecolors='none')
    ax2.set_facecolor('white')
    ax2.set_ylabel('R [mm]', fontsize=18)
    ax2.set_xlabel('Phi [rad]', fontsize=18)
    _configure_polar_axis(ax2, r_filtered)
    #ax2.set_title('Hit Distribution (R-Phi)',fontsize=20)
    if pcm is not None:
        plt.colorbar(pcm, ax=ax2, label='Hits/mm²')

    # 3. R-Z hit distribution
    ax3 = fig.add_subplot(gs[1, :])
    z_filtered = z_vals[geom_mask] if geom_mask.size else z_vals
    r_filtered_for_rz = r_filtered if geom_mask.size else r_vals
    hist, xedges, yedges = np.histogram2d(z_filtered, r_filtered_for_rz, bins=[100, 30])
    area_map_rz = compute_rz_area_map(layer_metadata, xedges, yedges)
    with np.errstate(invalid='ignore', divide='ignore'):
        hist_normalized = np.divide(hist, area_map_rz, where=area_map_rz > 0)
    masked_rz = np.ma.masked_where(hist_normalized <= 0, hist_normalized)
    cmap_rz = plt.cm.get_cmap('viridis').copy()
    cmap_rz.set_bad(color='white')
    pcm = ax3.pcolormesh(xedges[:-1], yedges[:-1], masked_rz.T, shading='auto', cmap=cmap_rz)
    ax3.set_xlabel('Z [mm]',fontsize=18)
    ax3.set_ylabel('R [mm]',fontsize=18)
    #ax3.set_title('Hit Distribution (R-Z)',fontsize=20)
    if pcm is not None:
        plt.colorbar(pcm, ax=ax3, label='Hits/mm²')
    


    fig.tight_layout()

    # Create occupancy-only figure
    fig_occ, ax_occ = plt.subplots(figsize=(7, 5))
    # Remove any substring like "(133 bunches)" while keeping other content
    compact_title = re.sub(r'\(\d+\s+bunches(?:/train)?\)', '', title_text)
    compact_title = re.sub(r'\s{2,}', ' ', compact_title).strip()
    fig_occ.suptitle(compact_title, fontsize=24)
    for label, vals, errs in series_data:
        ax_occ.errorbar(
            thresholds,
            vals,
            yerr=errs,
            fmt='o-',
            label=label,
            capsize=3
        )
    ax_occ.set_xlabel('Buffer depth', fontsize=18)
    ax_occ.set_ylabel('Layer occupancy', fontsize=18)
    ax_occ.set_yscale('log')
    _configure_log_yticks(ax_occ)
    ax_occ.set_xticks(thresholds)
    ax_occ.set_xticklabels([str(int(t)) for t in thresholds])
    ax_occ.grid(True)
    ax_occ.legend()
    fig_occ.tight_layout(rect=(0, 0, 1, 0.92))

    if output_prefix:
        fig.savefig(f'{output_prefix}_occupancy_analysis.png')
        fig.savefig(f'{output_prefix}_occupancy_analysis.pdf')
        fig_occ.savefig(f'{output_prefix}_occupancy_vs_buffer_only.png')
        fig_occ.savefig(f'{output_prefix}_occupancy_vs_buffer_only.pdf')
    
    plt.show()
    plt.close(fig)
    plt.close(fig_occ)


def plot_timing_analysis(stats, geometry_info, output_prefix=None, geometry_filter_tolerance=1.0):
    """Create timing visualizations with optional geometry-based hit filtering."""
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    plt.style.use(hep.style.CMS)
    # plt.rcParams.update({
    #     'font.family': 'serif',
    #     'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'serif'],
    #     'mathtext.fontset': 'dejavuserif',
    # })

    # Retrieve assumed time cut for occupancy calculation
    time_cut = stats['time_cut']
    print(f"Time cut: {time_cut}")

    # Retrieve assumed time cut for occupancy calculation
    time_cut = stats['time_cut']
    print(f"Time cut: {time_cut}")

    detector_label = _get_detector_display_name(geometry_info["detector_name"])
    title_text = f'Timing Analysis for {detector_label}'
    if time_cut > 0:
        title_text += f' (t<{time_cut} ns)'

    # Add the C^3 info and use rich text formatting for "Preliminary"
    #title_text += r'                   SiD_o2_v04@C$^{3}$-550 (266 bunches) - $\boldsymbol{\it{Preliminary}}$'
    title_text += r'                   C$^{3}$ 250 - Baseline (133 bunches) - $\boldsymbol{\it{Preliminary}}$'
    fig.suptitle(title_text, fontsize=24)

      
    gs = plt.GridSpec(2, 2)

    time_vals = ak.to_numpy(stats['times'])
    r_vals = ak.to_numpy(stats['positions']['r'])
    z_vals = ak.to_numpy(stats['positions']['z'])
    phi_vals = ak.to_numpy(stats['positions']['phi'])

    layer_metadata = extract_layer_geometry(geometry_info)
    geom_mask = filter_hits_to_geometry(r_vals, z_vals, layer_metadata,
                                        tolerance=geometry_filter_tolerance)
    if geom_mask.size:
        time_vals = time_vals[geom_mask]
        r_vals = r_vals[geom_mask]
        z_vals = z_vals[geom_mask]
        phi_vals = phi_vals[geom_mask]

    if time_vals.size == 0:
        print("JIM DEBUG [timing]", detector_label,
              "no hits after thresholds – skipping timing plot")
        return

    #t_edges = np.linspace(0, max(stats['times']), 100) #Jim: change time range
    t_edges = np.linspace(0, 100, 100)

    # 1. Timing distribution
    ax1 = fig.add_subplot(gs[0, 0])

    ax1.hist(time_vals, bins=100, range=(0, 100), histtype='step')
    ax1.set_xlabel('Time [ns]',fontsize=18)
    ax1.set_ylabel('Hits',fontsize=18)
    #ax1.set_title('Timing Distribution',fontsize=20)
    ax1.grid(True)

    # 2. Timing vs R
    ax2 = fig.add_subplot(gs[1, 0])

    if r_vals.size:
        r_edges = np.linspace(0, r_vals.max(), 21)
    else:
        r_edges = np.linspace(0, 1, 21)

    hist, xedges, yedges = np.histogram2d(r_vals,time_vals,
                            bins=[r_edges,t_edges]) 

    # Create a colormap that uses purple for zero values
    pcm = ax2.pcolormesh(xedges[:-1],yedges[:-1], hist.T, 
                        shading='auto', 
                        cmap='viridis',
                        vmin=0,
                        edgecolors='none')
    ax2.set_facecolor('#3F007D')
    
    ax2.set_ylabel('Time [ns]',fontsize=18)
    ax2.set_xlabel('R [mm]',fontsize=18)
    #ax2.set_title('Timing vs R')

    plt.colorbar(pcm, ax=ax2, label='Hits')




    # 3. Timing vs Z
    ax3 = fig.add_subplot(gs[1, 1])

    if z_vals.size:
        z_edges = np.linspace(z_vals.min(), z_vals.max(), 100)
    else:
        z_edges = np.linspace(-1, 1, 100)

    hist, xedges, yedges = np.histogram2d(z_vals, time_vals, 
                            bins=[z_edges, t_edges]) 

    # Create a colormap that uses purple for zero values
    pcm = ax3.pcolormesh(xedges[:-1], yedges[:-1], hist.T, 
                        shading='auto', 
                        cmap='viridis',
                        vmin=0,
                        edgecolors='none')
    ax3.set_facecolor('#3F007D')
    ax3.set_xlabel('Z [mm]',fontsize=18)
    ax3.set_ylabel('Time [ns]',fontsize=18)
    #ax3.set_title('Timing vs R')

    plt.colorbar(pcm, ax=ax3, label='Hits')


    # 4. Timing vs Phi
    ax4 = fig.add_subplot(gs[0, 1])

    phi_edges = np.linspace(-np.pi, np.pi, 51)

    hist, xedges, yedges = np.histogram2d(phi_vals, time_vals, 
                            bins=[phi_edges, t_edges]) 


    # Create a colormap that uses purple for zero values
    pcm = ax4.pcolormesh(xedges[:-1], yedges[:-1], hist.T, 
                        shading='auto', 
                        cmap='viridis',
                        vmin=0,
                        edgecolors='none')
    ax4.set_facecolor('#3F007D')
    ax4.set_xlabel('Phi [rad]',fontsize=18, labelpad=10)
    ax4.tick_params(axis='x', pad=8)
    ax4.set_ylabel('Time [ns]',fontsize=18)

    plt.colorbar(pcm, ax=ax4, label='Hits')

    plt.tight_layout()

    if output_prefix:
        plt.savefig(f'{output_prefix}_timing_analysis.png')
        plt.savefig(f'{output_prefix}_timing_analysis.pdf')
    plt.show()



   
def plot_detector_analysis(stats, geometry_info, detector_name, output_prefix=None,
                           geometry_filter_tolerance=1.0):
    """
    Create detailed visualizations of the detector analysis
    
    Parameters as before, with optional geometry_filter_tolerance allowing
    tighter or looser rejection of out-of-geometry hits (in mm).
    """
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)
    
    # 1. Occupancy vs threshold for each layer
    ax1 = fig.add_subplot(gs[0, 0])

    thresholds = sorted(stats['threshold_stats'].keys())
    for layer in sorted(geometry_info['layers'].keys()):
        occupancies = [stats['threshold_stats'][t]['per_layer'].get(layer, {'occupancy': 0})['occupancy'] * 100 
                      for t in thresholds]
        ax1.plot(thresholds, occupancies, 'o-', label=f'Layer {layer}')
    
    ax1.set_xlabel('Buffer depth',fontsize=18)
    ax1.set_ylabel('Occupancy (%)',fontsize=18)
    #ax1.set_title(f'{detector_name} Occupancy vs Hit Threshold by Layer',fontsize=20)
    ax1.set_xticks(thresholds)  # Set ticks to the actual threshold values
    ax1.set_xticklabels([str(int(t)) for t in thresholds])  # Format as integers
    ax1.grid(True)
    ax1.legend()
    
    # 2. R-Phi hit distribution
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')

    phi_vals = ak.to_numpy(stats['positions']['phi'])
    r_vals = ak.to_numpy(stats['positions']['r'])
    z_vals = ak.to_numpy(stats['positions']['z'])

    layer_metadata = extract_layer_geometry(geometry_info)
    geom_mask = filter_hits_to_geometry(r_vals, z_vals, layer_metadata,
                                        tolerance=geometry_filter_tolerance)
    phi_filtered = phi_vals[geom_mask] if geom_mask.size else phi_vals
    r_filtered = r_vals[geom_mask] if geom_mask.size else r_vals

    hist, _, _ = np.histogram2d(phi_filtered, r_filtered, bins=[50, 20])
    r_edge_source = r_filtered if geom_mask.size else r_vals
    if r_edge_source.size:
        r_edges = np.linspace(0, r_edge_source.max(), 21)
    else:
        r_edges = np.linspace(0, 1, 21)
    r_bin_width = np.diff(r_edges)[0]  # R bin width in mm
    phi_edges = np.linspace(-np.pi, np.pi, 51)
    R, PHI = np.meshgrid(r_edges[:-1], phi_edges[:-1])
    
    phi_bin_width = np.diff(phi_edges)[0]  # R bin width in mm
    r_bin_width = np.diff(r_edges)[0]  # R bin width in mm


    # Calculate bin sizes
    dphi = phi_edges[1] - phi_edges[0]
    dr = r_edges[1] - r_edges[0]

    r_centers = 0.5 * (r_edges[1:] + r_edges[:-1])

    bin_areas = np.zeros((len(phi_edges)-1, len(r_edges)-1))
    for j in range(len(r_centers)):
        bin_areas[:, j] = r_centers[j] * dr * dphi

    hist_normalized = hist / (bin_areas + 1e-10)

        

    # Calculate bin area (rad × ns)
    bin_area = phi_bin_width * phi_bin_width
    density = hist.T / bin_area

    pcm = ax2.pcolormesh(PHI, R, hist_normalized.T, shading='auto')
    #ax2.set_title(f'{detector_name} Hit Distribution (R-Phi)',fontsize=20)
    ax2.set_ylabel('R [mm]', fontsize=18)
    ax2.set_xlabel('Phi [rad]', fontsize=18)
    _configure_polar_axis(ax2, r_filtered)
    plt.colorbar(pcm, ax=ax2, label='Hits/mm²')
    
    # ax2.text(0.02, 0.02, f"Bin size: {phi_bin_width:.2f} rad × {t_bin_width:.1f} ns", 
    #         transform=ax4.transAxes, bbox=dict(facecolor='white', alpha=0.7))

    # 3. R-Z hit distribution
    ax3 = fig.add_subplot(gs[1, :])

    z_filtered = z_vals[geom_mask] if geom_mask.size else z_vals
    r_filtered_for_rz = r_filtered if geom_mask.size else r_vals
    hist, xedges, yedges = np.histogram2d(z_filtered, r_filtered_for_rz, bins=[100, 20])
    

    # Calculate bin sizes for normalization
    dz = xedges[1] - xedges[0]
    dr = yedges[1] - yedges[0]

    bin_area = dz * dr
    print("dz = ",dz,"mm,  dr = ",dr,"mm ,  bin_area = ",bin_area,"mm²")
    hist_normalized = hist / bin_area

    pcm = ax3.pcolormesh(xedges[:-1], yedges[:-1], hist_normalized.T, shading='auto')
    ax3.set_xlabel('Z [mm]',fontsize=18)
    ax3.set_ylabel('R [mm]',fontsize=18)
    #ax3.set_title(f'{detector_name} Hit Distribution (R-Z)',fontsize=20)
    plt.colorbar(pcm, ax=ax3, label='Hits/mm²')
    
    plt.tight_layout()
    
    if output_prefix:
        plt.savefig(f'{output_prefix}_{detector_name}_analysis.png')
        plt.savefig(f'{output_prefix}_{detector_name}_analysis.pdf')
    plt.show()
                
                
def print_occupancy_statistics(results, geometry_info):
    """
    Print detailed occupancy statistics
    """
    print("\nDetailed occupancy statistics:")
    
    for threshold in sorted(results['threshold_stats'].keys()):
        print(f"\nThreshold = {threshold}:")
        stats = results['threshold_stats'][threshold]
        
        for layer in sorted(stats['per_layer'].keys()):
            layer_stats = stats['per_layer'][layer]
            print(f"\nLayer {layer}:")
            print(f"  Cells hit: {layer_stats['cells_hit']}")
            print(f"  Total hits: {layer_stats['total_hits']}")
            print(f"  Occupancy: {layer_stats['occupancy']*100:.10f}%")
            print(f"  Mean hits per hit cell: {layer_stats['mean_hits']:.10f}")




def analyze_detectors_and_plot(DETECTOR_CONFIGS=None, detectors_to_analyze=None, event_trees=None,
                              main_xml=None, remove_zeros=True, time_cut=-1, 
                              calo_hit_time_def=0, energy_thresholds=None, nlayer_batch=1):
    """
    Analyze detectors and create plots
    
    Parameters:
    -----------
    DETECTOR_CONFIGS : dict
        Dictionary of detector configurations
    detectors_to_analyze : list
        List of (detector_name, xml_file) tuples
    event_trees : list
        List of uproot event trees
    main_xml : str
        Path to main XML file
    remove_zeros : bool
        Whether to remove hits with zero positions
    time_cut : float
        Cut on hit time in ns (-1 for no cut)
    calo_hit_time_def : int
        0: use min time of contributions, 1: time when cumulative energy exceeds threshold
    energy_thresholds : dict, optional
        Dictionary of energy thresholds for different detector types (see analyze_detector_hits)
    nlayer_batch : int
        Number of layers to group together in occupancy plots
    """
    # Default thresholds
    if energy_thresholds is None:
        energy_thresholds = {
            'silicon': 30e-3,               # 30 keV for silicon
            'ecal_hits': 5e-3,              # 5 MeV for ECAL hits
            'ecal_contributions': 0.2e-3,    # 200 keV for ECAL contributions
            'hcal_hits': 20e-3,             # 20 MeV for HCAL hits
            'hcal_contributions': 1e-3,      # 1 MeV for HCAL contributions
            'muon_hits': 50e-3,             # 50 MeV for Muon hits
            'muon_contributions': 5e-3       # 5 MeV for Muon contributions
        }

    print("\nAnalyzing detectors:")
    for detector_name, xml_file in detectors_to_analyze:
        print(f"\nAnalyzing {detector_name}...")
        try:
            detector_config = DETECTOR_CONFIGS[detector_name]
            xmls = get_xmls()
            main_xml = xmls['main_xml']
            # Pass detector name to get specific debug info
            constants = parse_detector_constants(main_xml, detector_name, detector_xml_file=xml_file)
            geometry_info = get_geometry_info(xml_file, detector_config, constants=constants)
            
            # Print geometry info
            print(f"\nGeometry info for {detector_name}:")
            print(f"Total cells: {geometry_info['total_cells']}")
            for layer, info in sorted(geometry_info['layers'].items()):
                print(f"\nLayer {layer}:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
            
    
            # Define thresholds to analyze
            #hit_thresholds = [1, 2, 3, 4, 5,6,7,8,9, 10]
            hit_thresholds = [1, 2,3,4]
            stats = analyze_detector_hits(event_trees,detector_name=detector_name,
                                            config=detector_config,
                                            hit_thresholds=hit_thresholds,
                                            geometry_file=xml_file,
                                            constants=constants,
                                            main_xml=main_xml,
                                            remove_zeros=remove_zeros,
                                            time_cut=time_cut,
                                            calo_hit_time_def=calo_hit_time_def,
                                            energy_thresholds=energy_thresholds)
            
            # Create visualizations
            plot_occupancy_analysis(stats, geometry_info, output_prefix=detector_name,time_cut=time_cut,nlayer_batch=nlayer_batch)
            
            plot_timing_analysis(stats, geometry_info, output_prefix=detector_name)

            # Print detailed statistics for threshold=1
            for threshold in range(1, 10):  # Loop from 1 to 5 (inclusive)
                print(f"\nDetailed occupancy statistics (threshold={threshold}):")

                print(stats['threshold_stats'])
                
                # for layer, info in sorted(stats['threshold_stats'][threshold]['per_layer'].items()):
                #     if layer in geometry_info['layers']:
                #         print(f"\nLayer {layer}:")
                #         #print(f"  Cells hit: {info['cells_hit']}")
                #         print(f"  Cells hit: {info['pixels_hit']}")

                #         print(f"  Total hits: {info['total_hits']}")
                #         print(f"  Occupancy: {info['occupancy']*100:.10f}%")
                #         print(f"  Mean hits per hit cell: {info['mean_hits']:.10f}")
            
                    
        except Exception as e:
            print(f"Error processing {detector_name}: {str(e)}")
            traceback.print_exc()

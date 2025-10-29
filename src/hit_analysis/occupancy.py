import awkward as ak
import numpy as np
from collections import Counter

from src.geometry_parsing.k4geo_parsers import find_matching_ring,calculate_module_position
from src.segmentation.pixelizers import calculate_local_coordinates,get_pixel_indices,get_pixel_id
from src.geometry_parsing.geometry_info import get_geometry_info
from src.geometry_parsing.cellid_decoders import decode_dd4hep_cellid

# Import vectorized hit processing
from src.utils.vectorized_hit_processing import process_hits_vectorized, benchmark_hit_processing




def process_hit(hit_data, detector_name, layer_info, config):
    """
    Process a single hit to determine its pixel/cell location.
    
    Parameters:
    -----------
    hit_data : dict
        {'pos': (x,y,z), 'cellid_decoded': decoded_cellid}
    detector_name : str
        Name of the detector
    layer_info : dict
        Layer geometry information
    config : DetectorConfig
        Detector configuration
        
    Returns:
    --------
    tuple: pixel key for hit counting
    """
    decoded = hit_data['cellid_decoded']
    hit_pos = hit_data['pos']
    layer = decoded['layer']
    
    # Handle forward calorimeters directly
    if detector_name in ['BeamCal', 'LumiCal']:
        return (layer, decoded['x'], decoded['y'])
    
    # Get module number from decoded cellID
    module = decoded['module']
    
    # Determine if this is an endcap by checking for rings
    is_endcap = 'rings' in layer_info
    
    try:
        if is_endcap:
            # Find which ring this hit belongs to
            ring_info = find_matching_ring(hit_pos, layer_info)
            if ring_info is None:
                print(f"Warning: Hit at {hit_pos} doesn't match any ring in layer {layer}")
                return (layer, module)
                
            # Calculate local coordinates within ring module
            module_pos = calculate_module_position(ring_info, module, 'endcap')
            if module_pos is None:
                return (layer, module)
                
            local_coords = calculate_local_coordinates(
                hit_pos, module_pos, module_pos[3], 'endcap'
            )
            
            # Get pixel indices using ring module dimensions
            if ring_info['type'] == 'trd':
                # For trapezoid, use average width
                module_dims = (ring_info['width'], ring_info['length'])
            else:
                module_dims = (ring_info['width'], ring_info['length'])
                
            pixel_indices = get_pixel_indices(
                local_coords, module_dims, config.get_cell_size(layer=layer)
            )
            
            # Include ring number in pixel key for endcap
            ring_idx = layer_info['rings'].index(ring_info)
            return (layer, ring_idx, module, pixel_indices[0], pixel_indices[1])
            
        else:
            # Barrel detector
            if 'rc' not in layer_info:
                # Try to calculate rc from inner/outer radii
                if 'inner_r' in layer_info and 'outer_r' in layer_info:
                    layer_info['rc'] = (layer_info['inner_r'] + layer_info['outer_r']) / 2
                else:
                    print(f"Warning: Cannot determine radius for layer {layer}")
                    return (layer, module)
            
            # Calculate module position and local coordinates
            module_pos = calculate_module_position(layer_info, module, 'barrel')
            if module_pos is None:
                return (layer, module)
                
            local_coords = calculate_local_coordinates(
                hit_pos, module_pos, module_pos[3], 'barrel'
            )
            
            # Get module dimensions
            if all(key in layer_info for key in ['module_width', 'module_length']):
                module_dims = (layer_info['module_width'], layer_info['module_length'])
            else:
                print(f"Warning: Missing module dimensions for layer {layer}")
                return (layer, module)
            
            # Calculate pixel indices
            pixel_indices = get_pixel_indices(
                local_coords, module_dims, config.get_cell_size(layer=layer)
            )
            
            return (layer, module, pixel_indices[0], pixel_indices[1])
            
    except Exception as e:
        print(f"Warning: Error processing hit in {detector_name}, layer {layer}: {str(e)}")
        return (layer, module)




















def calculate_layer_occupancy(layer_cells, total_cells, threshold):
    """
    Calculate occupancy for a layer at given threshold
    
    Parameters:
    -----------
    layer_cells : dict
        Dictionary of cellID -> hit count for cells in this layer
    total_cells : int
        Total number of cells in layer from geometry
    threshold : int
        Hit threshold value
        
    Returns:
    --------
    float : Fraction of cells with hits ≥ threshold
    """
    if total_cells == 0:
        return 0.0
    cells_above = sum(1 for hits in layer_cells.values() if hits >= threshold)
    return cells_above / total_cells



def _resolve_energy_thresholds(detector_name: str, detector_class: str, energy_thresholds: dict):
    """Resolve per-detector energy thresholds with barrel/endcap overrides.

    Resolution order per quantity (hits/contributions/silicon):
      1) Exact detector name override, e.g. 'ECalEndcap_hits', 'SiTrackerBarrel_silicon'
      2) Class-level default (ecal/hcal/muon/silicon/forward calo)

    Returns a dict with keys: 'si', 'hits', 'contrib'. Missing quantities are 0.
    """
    det_lower = (detector_class or "").lower()

    # Class defaults
    if det_lower in ['vertex', 'tracker']:
        base_si = energy_thresholds.get('silicon', 30e-3)
        base_hits = 0
        base_contrib = 0
    elif det_lower == 'ecal':
        base_si = 0
        base_hits = energy_thresholds.get('ecal_hits', 5e-3)
        base_contrib = energy_thresholds.get('ecal_contributions', 0.2e-3)
    elif det_lower == 'hcal':
        base_si = 0
        base_hits = energy_thresholds.get('hcal_hits', 20e-3)
        base_contrib = energy_thresholds.get('hcal_contributions', 1e-3)
    elif det_lower == 'muon':
        base_si = 0
        base_hits = energy_thresholds.get('muon_hits', 50e-3)
        base_contrib = energy_thresholds.get('muon_contributions', 5e-3)
    elif det_lower in ['beamcal', 'lumical']:
        # Use ECAL-like defaults for forward calorimeters
        base_si = 0
        base_hits = energy_thresholds.get('ecal_hits', 5e-3)
        base_contrib = energy_thresholds.get('ecal_contributions', 0.2e-3)
    else:
        base_si = 0
        base_hits = 0
        base_contrib = 0

    # Exact-name overrides (e.g. 'ECalEndcap_hits', 'SiTrackerBarrel_silicon')
    si = energy_thresholds.get(f'{detector_name}_silicon', base_si)
    hits = energy_thresholds.get(f'{detector_name}_hits', base_hits)
    contrib = energy_thresholds.get(f'{detector_name}_contributions', base_contrib)

    return {'si': si, 'hits': hits, 'contrib': contrib}


def analyze_detector_hits(events_trees, detector_name, config, buffer_depths=None, 
                         geometry_file=None, constants=None, main_xml=None, remove_zeros=True, 
                         time_cut=-1, calo_hit_time_def=0,
                         energy_thresholds=None):

    """
    Analyze hits with improved coordinate and timing handling for different detector types.
    
    Parameters:
    -----------
    events_trees : list
        List of uproot event trees
    detector_name : str
        Name of the detector
    config : DetectorConfig
        Detector configuration
    buffer_depths : list, optional
        List of buffer depths for occupancy calculation (applied to counts-per-cell)
    geometry_file : str, optional
        Path to geometry XML file
    constants : dict, optional
        Constants dictionary
    main_xml : str, optional
        Path to main XML file
    remove_zeros : bool
        Whether to remove hits with zero positions
    time_cut : float
        Cut on hit time in ns (-1 for no cut)
    calo_hit_time_def : int
        0: use min time of contributions, 1: time when cumulative energy exceeds threshold
    energy_thresholds : dict, optional
        Dictionary of energy thresholds for different detector types:
        {
            'silicon': 30e-3,              # For silicon detectors (vertex, tracker), in GeV
            'ecal_hits': 5e-3,             # For ECAL hits (cumulative energy), in GeV
            'ecal_contributions': 0.2e-3,   # For ECAL hit contributions, in GeV
            'hcal_hits': 20e-3,            # For HCAL hits (cumulative energy), in GeV
            'hcal_contributions': 1e-3,     # For HCAL hit contributions, in GeV
            'muon_hits': 50e-3,            # For Muon system hits, in GeV
            'muon_contributions': 5e-3      # For Muon hit contributions, in GeV
        }
    """
    if buffer_depths is None:
        buffer_depths = [1]
    
    # Default energy thresholds if none provided
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


    # Determine detector type
    det_class = config.detector_class
    is_silicon = det_class.lower() in ['vertex', 'tracker']
    is_ecal = det_class.lower() == 'ecal'
    is_hcal = det_class.lower() == 'hcal'
    is_muon = det_class.lower() == 'muon'
    is_forward_calo = det_class.lower() in ['beamcal', 'lumical']

    # Resolve thresholds with per-detector overrides
    _thr = _resolve_energy_thresholds(detector_name, det_class, energy_thresholds)
    si_threshold = _thr['si'] if is_silicon else 0
    hits_threshold = _thr['hits'] if (is_ecal or is_hcal or is_muon or is_forward_calo) else 0
    contributions_threshold = _thr['contrib'] if (is_ecal or is_hcal or is_muon or is_forward_calo) else 0


    # Build branch names
    hits_prefix = f"{detector_name}Hits"
    hits_contrib_prefix = f"{detector_name}HitsContributions"


    cellid_branch = f"{hits_prefix}/{hits_prefix}.cellID"
    pos_branches = [f"{hits_prefix}/{hits_prefix}.position.x",
                    f"{hits_prefix}/{hits_prefix}.position.y",
                    f"{hits_prefix}/{hits_prefix}.position.z"]
    

    # Include energy branch for Silicon detectors for threshold application
    if is_silicon:
        energy_branch = f"{hits_prefix}/{hits_prefix}.eDep"
        time_branch = f"{hits_prefix}/{hits_prefix}.time"
        main_branches = [cellid_branch] + pos_branches + [energy_branch, time_branch]
    elif is_ecal or is_hcal or is_muon or is_forward_calo:
        # For calorimeters, we need contributions
        energy_branch = f"{hits_prefix}/{hits_prefix}.energy"
        contrib_time_branch = f"{hits_contrib_prefix}/{hits_contrib_prefix}.time"
        contrib_energy_branch = f"{hits_contrib_prefix}/{hits_contrib_prefix}.energy"
        contrib_begin_branch = f"{hits_prefix}/{hits_prefix}.contributions_begin"
        contrib_end_branch = f"{hits_prefix}/{hits_prefix}.contributions_end"
        main_branches = [cellid_branch] + pos_branches + [
            energy_branch, contrib_time_branch, contrib_energy_branch, 
            contrib_begin_branch, contrib_end_branch
        ]
    else:
        # Fallback for unknown detector types
        time_branch = f"{hits_prefix}/{hits_prefix}.time"
        main_branches = [cellid_branch] + pos_branches + [time_branch]


    # Read event arrays
    try:

        filtered_cellids = []
        filtered_x = []
        filtered_y = []
        filtered_z = []
        filtered_t = []
        filtered_e = []  # Store energy values for later use



        for events_tree in events_trees:
            #branches_to_read = [cellid_branch] + pos_branches + time_branches
            #array = events_tree.arrays(branches_to_read)
            try:
                array = events_tree.arrays(main_branches)
            except Exception as e:
                # Missing collection(s) in this tree; skip gracefully
                print(f"Error reading event data for {detector_name}: {e}")
                # Append empty arrays to keep alignment
                filtered_cellids.append(ak.Array([]))
                filtered_x.append(ak.Array([]))
                filtered_y.append(ak.Array([]))
                filtered_z.append(ak.Array([]))
                filtered_t.append(ak.Array([]))
                filtered_e.append(ak.Array([]))
                continue

            cellids = array[cellid_branch]
            x = array[pos_branches[0]]
            y = array[pos_branches[1]]
            z = array[pos_branches[2]]

            # Handle different detector types
            if is_silicon:
                time = array[time_branch]
                energy = array[energy_branch]
                
            # Handle different detector types
            if is_silicon:
                time = array[time_branch]
                energy = array[energy_branch]  # This is eDep
                
                # Apply silicon energy threshold if specified
                if si_threshold > 0:
                    energy_mask = energy >= si_threshold
                    cellids = cellids[energy_mask]
                    x = x[energy_mask]
                    y = y[energy_mask]
                    z = z[energy_mask]
                    time = time[energy_mask]
                    energy = energy[energy_mask]
                
            elif is_ecal or is_hcal or is_muon or is_forward_calo:
                energy = array[energy_branch]
                contrib_times = array[contrib_time_branch]
                contrib_energies = array[contrib_energy_branch]
                contrib_begin = array[contrib_begin_branch]
                contrib_end = array[contrib_end_branch]
                
                
                # Process each event to extract hit times based on contributions
                hit_times = []
                hit_energies = []
                valid_hit_mask = []
                
                for event_idx in range(len(cellids)):
                    event_times = []
                    event_cumul_energies = []
                    event_valid_hits = []
                    
                    for hit_idx in range(len(cellids[event_idx])):
                        # Get contribution indices for this hit
                        begin_idx = contrib_begin[event_idx][hit_idx]
                        end_idx = contrib_end[event_idx][hit_idx]
                        
                        if begin_idx < end_idx:
                            # Extract contributions for this hit
                            hit_contrib_times = contrib_times[event_idx][begin_idx:end_idx]
                            hit_contrib_energies = contrib_energies[event_idx][begin_idx:end_idx]

                            # Apply energy threshold to contributions if specified
                            if contributions_threshold > 0:
                                energy_mask = hit_contrib_energies >= contributions_threshold
                                if ak.any(energy_mask):
                                    hit_contrib_times = hit_contrib_times[energy_mask]
                                    hit_contrib_energies = hit_contrib_energies[energy_mask]
                                else:
                                    # No contributions pass the threshold
                                    event_times.append(float('inf'))
                                    event_cumul_energies.append(0)
                                    event_valid_hits.append(False)
                                    continue
                            
                            # Sort contributions by time for cumulative energy calculation
                            sort_indices = ak.argsort(hit_contrib_times)
                            sorted_times = hit_contrib_times[sort_indices]
                            sorted_energies = hit_contrib_energies[sort_indices]
                            
                            if calo_hit_time_def == 1 and hits_threshold > 0:
                                # Find time when cumulative energy exceeds threshold
                                cumul_energy = 0
                                threshold_time = float('inf')
                                
                                for t_idx in range(len(sorted_times)):
                                    cumul_energy += sorted_energies[t_idx]
                                    if cumul_energy >= hits_threshold:
                                        threshold_time = sorted_times[t_idx]
                                        break
                                
                                event_times.append(threshold_time)
                                event_cumul_energies.append(cumul_energy)
                                event_valid_hits.append(cumul_energy >= hits_threshold)
                            else:
                                # Use minimum time (earliest contribution)
                                min_time = ak.min(hit_contrib_times) if len(hit_contrib_times) > 0 else float('inf')
                                total_energy = ak.sum(hit_contrib_energies)
                                
                                event_times.append(min_time)
                                event_cumul_energies.append(total_energy)
                                event_valid_hits.append(total_energy >= hits_threshold)
                        else:
                            # No contributions, use large values
                            event_times.append(float('inf'))
                            event_cumul_energies.append(0)
                            event_valid_hits.append(False)
                    
                    hit_times.append(event_times)
                    hit_energies.append(event_cumul_energies)
                    valid_hit_mask.append(event_valid_hits)
                
                time = ak.Array(hit_times)
                energy = ak.Array(hit_energies)
                
                # Apply calorimeter energy threshold if specified
                if hits_threshold > 0:
                    valid_mask = ak.Array(valid_hit_mask)
                    cellids = cellids[valid_mask]
                    x = x[valid_mask]
                    y = y[valid_mask]
                    z = z[valid_mask]
                    time = time[valid_mask]
                    energy = energy[valid_mask]
            else:
                # For unknown detector types, use simple time 
                time = array[time_branch]
                energy = ak.ones_like(time)  # Default energy



            # Apply filtering: Keep events that contain at least one non-zero hit
            if remove_zeros:
                non_zero_mask = ak.any((x != 0.0) | (y != 0.0) | (z != 0.0), axis=1)
                cellids = cellids[non_zero_mask]
                x = x[non_zero_mask]
                y = y[non_zero_mask]
                z = z[non_zero_mask]
                time = time[non_zero_mask]
                energy = energy[non_zero_mask]

            # time cut
            if time_cut > 0:
                
                print("Analysis occupancy for time_cut = ", time_cut)

                time_mask = time < time_cut

                cellids = cellids[time_mask]
                x = x[time_mask]
                y = y[time_mask]
                z = z[time_mask]
                time = time[time_mask]
                energy = energy[time_mask]

            # Flatten arrays and store them
            filtered_cellids.append(ak.flatten(cellids))
            filtered_x.append(ak.flatten(x))
            filtered_y.append(ak.flatten(y))
            filtered_z.append(ak.flatten(z))
            filtered_t.append(ak.flatten(time))
            filtered_e.append(ak.flatten(energy))

        # Concatenate filtered results across all events (handle all-empty case)
        if filtered_cellids:
            non_empty_idx = [i for i, arr in enumerate(filtered_cellids) if len(arr) > 0]
        else:
            non_empty_idx = []
        if non_empty_idx:
            cellids_combined = ak.concatenate([filtered_cellids[i] for i in non_empty_idx])
            x_flat = ak.concatenate([filtered_x[i] for i in non_empty_idx])
            y_flat = ak.concatenate([filtered_y[i] for i in non_empty_idx])
            z_flat = ak.concatenate([filtered_z[i] for i in non_empty_idx])
            t_flat = ak.concatenate([filtered_t[i] for i in non_empty_idx])
            e_flat = ak.concatenate([filtered_e[i] for i in non_empty_idx])
        else:
            # No hits found; return an empty-but-well-formed result
            cellids_combined = ak.Array([])
            x_flat = ak.Array([])
            y_flat = ak.Array([])
            z_flat = ak.Array([])
            t_flat = ak.Array([])
            e_flat = ak.Array([])

    except Exception as e:
        print(f"Error reading event data for {detector_name}: {e}")
        return None
    
    # Get geometry info if available
    geometry_info = None
    if geometry_file:
        try:
            geometry_info = get_geometry_info(geometry_file, config, constants, main_xml)
        except Exception as e:
            print(f"Error processing {detector_name}: {e}")
            return None
    
    pixel_hits = {}
    # Use "config" (not detector_config)
    for cellid, hit_x, hit_y, hit_z in zip(cellids_combined, x_flat, y_flat, z_flat):
        if remove_zeros and hit_x == 0 and hit_y == 0 and hit_z == 0:
            continue  # Skip zero-position hits
    
        try:
            decoded = decode_dd4hep_cellid(cellid, detector_name)
            # Call our unified function; note we pass config (not detector_config)
            pixel_key = get_pixel_id((hit_x, hit_y, hit_z), decoded,
                                     config, geometry_info,
                                     effective_cell_size=None, verbose=True)
            pixel_hits[pixel_key] = pixel_hits.get(pixel_key, 0) + 1
        except Exception as e:
            print(f"Warning: Error processing hit in {detector_name}: {e}")
            continue
    
    # Compute per-threshold statistics
    stats = {}
    for threshold in buffer_depths:
        layer_stats = {}
        for pixel_key, count in pixel_hits.items():
            layer = pixel_key[0]
            if layer not in layer_stats:
                layer_stats[layer] = {
                    'cells_hit': 0,
                    'total_hits': 0,
                    'cells_above_threshold': 0,
                    'max_hits': 0,
                    'hit_counts': []
                }
            layer_stats[layer]['cells_hit'] += 1
            layer_stats[layer]['total_hits'] += count
            if count >= threshold:
                layer_stats[layer]['cells_above_threshold'] += 1
            layer_stats[layer]['max_hits'] = max(layer_stats[layer]['max_hits'], count)
            layer_stats[layer]['hit_counts'].append(count)
        for layer, lstats in layer_stats.items():
            if geometry_info and layer in geometry_info['layers']:
                total_cells = geometry_info['layers'][layer].get('total_cells', 0)
                lstats['occupancy'] = (lstats['cells_above_threshold'] / total_cells) if total_cells > 0 else 0.0
            else:
                lstats['occupancy'] = 0.0
            lstats['mean_hits'] = np.mean(lstats['hit_counts'])
            del lstats['hit_counts']
        # Build safe stats even when layer_stats is empty
        if layer_stats:
            max_hits_per_cell = max(s['max_hits'] for s in layer_stats.values())
            overall_cells_hit = sum(s['cells_hit'] for s in layer_stats.values())
            overall_cells_above = sum(s['cells_above_threshold'] for s in layer_stats.values())
        else:
            max_hits_per_cell = 0
            overall_cells_hit = 0
            overall_cells_above = 0
        stats[threshold] = {
            'overall_cells_hit': overall_cells_hit,
            'overall_cells_above_threshold': overall_cells_above,
            'max_hits_per_cell': max_hits_per_cell,
            'hit_distribution': dict(Counter(count for count in pixel_hits.values())),
            'per_layer': layer_stats
        }
    
    return {
        'detector_name': detector_name,
        'detector_class': config.detector_class,
        'time_cut': time_cut,
        'energy_threshold': {
            # Return the specific threshold used for this detector type
            'hit_threshold': hits_threshold if (is_ecal or is_hcal or is_muon or is_forward_calo) else 0,
            'contribution_threshold': contributions_threshold if (is_ecal or is_hcal or is_muon or is_forward_calo) else 0,
            'silicon': si_threshold if is_silicon else 0,
            # Also include the full threshold dictionary for reference
            'thresholds_dict': energy_thresholds
        },
        'calo_time_definition': "cumulative_energy" if calo_hit_time_def == 1 else "minimum",
        'times': t_flat,
        'energies': e_flat,
        'threshold_stats': stats,
        'positions': {
            'r': np.sqrt(x_flat**2 + y_flat**2),
            'phi': np.arctan2(y_flat, x_flat),
            'z': z_flat
        }
    }


def analyze_detector_hits_vectorized(events_trees, detector_name, config, buffer_depths=None, 
                                   geometry_file=None, constants=None, main_xml=None, 
                                   remove_zeros=True, time_cut=-1, calo_hit_time_def=0,
                                   energy_thresholds=None, use_vectorized=True):
    """
    Analyze hits using vectorized processing for improved performance.
    
    This is an optimized version of analyze_detector_hits that uses vectorized
    operations instead of Python loops for significant speed improvements.
    
    Parameters are the same as analyze_detector_hits, with additional:
    -----------
    use_vectorized : bool
        If True, use vectorized hit processing. If False, fall back to original method.
    """
    
    print(f"Analyzing {detector_name} with {'vectorized' if use_vectorized else 'traditional'} processing...")
    
    # First, do all the data loading and filtering (same as original)
    # This part is already reasonably optimized with awkward arrays
    
    # Use the original function for data loading and preprocessing
    result = analyze_detector_hits(
        events_trees, detector_name, config, buffer_depths,
        geometry_file, constants, main_xml, remove_zeros, 
        time_cut, calo_hit_time_def, energy_thresholds
    )
    
    if not use_vectorized or result is None:
        return result
    
    #Dimitris!!! temporary override, return the result of the original function
    #print("!!! Cautious override, returning the result of the original function !!!")
    #return result
    
    # Extract the flattened hit data that was processed in the original function
    # We'll re-process the pixel hit counting part with vectorized operations
    
    try:
        # Re-read the data for vectorized processing
        if buffer_depths is None:
            buffer_depths = [1]
            
        # Determine detector type (same logic as original)
        is_silicon = config.detector_class.lower() in ['vertex', 'tracker']
        is_ecal = config.detector_class.lower() == 'ecal'
        is_hcal = config.detector_class.lower() == 'hcal'
        is_muon = config.detector_class.lower() == 'muon'
        is_forward_calo = config.detector_class.lower() in ['beamcal', 'lumical']
        
        # Set energy thresholds (same logic as original)
        if energy_thresholds is None:
            energy_thresholds = {
                'silicon': 30e-3, 'ecal_hits': 5e-3, 'ecal_contributions': 0.2e-3,
                'hcal_hits': 20e-3, 'hcal_contributions': 1e-3,
                'muon_hits': 50e-3, 'muon_contributions': 5e-3
            }
        
        # Build branch names (same as original)
        hits_prefix = f"{detector_name}Hits"
        cellid_branch = f"{hits_prefix}/{hits_prefix}.cellID"
        pos_branches = [f"{hits_prefix}/{hits_prefix}.position.x",
                       f"{hits_prefix}/{hits_prefix}.position.y",
                       f"{hits_prefix}/{hits_prefix}.position.z"]
        
        if is_silicon:
            energy_branch = f"{hits_prefix}/{hits_prefix}.eDep"
            time_branch = f"{hits_prefix}/{hits_prefix}.time"
            main_branches = [cellid_branch] + pos_branches + [energy_branch, time_branch]
        else:
            # For calorimeters, use the complex branch structure from original
            energy_branch = f"{hits_prefix}/{hits_prefix}.energy"
            contrib_time_branch = f"{detector_name}HitsContributions/{detector_name}HitsContributions.time"
            contrib_energy_branch = f"{detector_name}HitsContributions/{detector_name}HitsContributions.energy"
            contrib_begin_branch = f"{hits_prefix}/{hits_prefix}.contributions_begin"
            contrib_end_branch = f"{hits_prefix}/{hits_prefix}.contributions_end"
            main_branches = [cellid_branch] + pos_branches + [
                energy_branch, contrib_time_branch, contrib_energy_branch, 
                contrib_begin_branch, contrib_end_branch
            ]
        
        # Read and process data (simplified version - focus on the vectorizable part)
        filtered_cellids = []
        filtered_x = []
        filtered_y = []
        filtered_z = []
        
        for events_tree in events_trees:
            array = events_tree.arrays(main_branches)
            
            cellids = array[cellid_branch]
            x = array[pos_branches[0]]
            y = array[pos_branches[1]]
            z = array[pos_branches[2]]
            
            # Apply energy and time filtering (same logic as original but simplified)
            if is_silicon:
                time = array[time_branch]
                energy = array[energy_branch]
                
                # Apply thresholds
                _thr = _resolve_energy_thresholds(detector_name, config.detector_class, energy_thresholds)
                si_threshold = _thr['si']
                if si_threshold > 0:
                    energy_mask = energy >= si_threshold
                    cellids = cellids[energy_mask]
                    x = x[energy_mask]
                    y = y[energy_mask]
                    z = z[energy_mask]
                    time = time[energy_mask]
                
                # Apply time cut
                if time_cut > 0:
                    time_mask = time < time_cut
                    cellids = cellids[time_mask]
                    x = x[time_mask]
                    y = y[time_mask]
                    z = z[time_mask]
            
            # Apply zero filtering
            if remove_zeros:
                non_zero_mask = ak.any((x != 0.0) | (y != 0.0) | (z != 0.0), axis=1)
                cellids = cellids[non_zero_mask]
                x = x[non_zero_mask]
                y = y[non_zero_mask]
                z = z[non_zero_mask]
            
            # Flatten and store
            filtered_cellids.append(ak.flatten(cellids))
            filtered_x.append(ak.flatten(x))
            filtered_y.append(ak.flatten(y))
            filtered_z.append(ak.flatten(z))
        
        # Combine all data
        cellids_combined = ak.concatenate(filtered_cellids)
        x_flat = ak.concatenate(filtered_x)
        y_flat = ak.concatenate(filtered_y)
        z_flat = ak.concatenate(filtered_z)
        
        print(f"Processing {len(cellids_combined)} hits with vectorized operations...")
        
        # Get geometry info
        geometry_info = None
        if geometry_file:
            from src.geometry_parsing.geometry_info import get_geometry_info
            geometry_info = get_geometry_info(geometry_file, config, constants, main_xml)
        
        # Use vectorized processing for the hit counting and analysis
        cellids_np = ak.to_numpy(cellids_combined)
        x_np = ak.to_numpy(x_flat)
        y_np = ak.to_numpy(y_flat)
        z_np = ak.to_numpy(z_flat)

        vectorized_stats = process_hits_vectorized(
            cellids_np, x_np, y_np, z_np,
            detector_name, buffer_depths, geometry_info, config
        )
        
        # Update the result with vectorized statistics
        result['threshold_stats'] = vectorized_stats
        result['vectorized_processing'] = True
        
        return result
        
    except Exception as e:
        print(f"Error in vectorized processing for {detector_name}: {e}")
        print("Falling back to traditional processing...")
        return result  # Return original result as fallback


    




def analyze_vertex_detector(events_trees, hit_thresholds=None, geometry_file=None):
    """
    Comprehensive analysis of SiVertexBarrel hits
    
    Parameters:
    -----------
    events_trees : list of uproot.TTree
        List of event trees from uproot
    hit_thresholds : list of int, optional
        List of hit thresholds to analyze occupancy
    geometry_file : str, optional
        Path to geometry XML file
        
    Returns:
    --------
    dict with analysis results
    """
    if hit_thresholds is None:
        hit_thresholds = [1]
        
    # Get cellIDs and positions
    arrays = [
        events_tree.arrays([
            'SiVertexBarrelHits/SiVertexBarrelHits.cellID',
            'SiVertexBarrelHits/SiVertexBarrelHits.position.x',
            'SiVertexBarrelHits/SiVertexBarrelHits.position.y',
            'SiVertexBarrelHits/SiVertexBarrelHits.position.z'
        ])
        for events_tree in events_trees
    ]
    
    # Combine all hits
    cellids = [array['SiVertexBarrelHits/SiVertexBarrelHits.cellID'] for array in arrays]
    cellids_combined = ak.concatenate([ak.flatten(array) for array in cellids])
    
    x = ak.concatenate([ak.flatten(array['SiVertexBarrelHits/SiVertexBarrelHits.position.x']) for array in arrays])
    y = ak.concatenate([ak.flatten(array['SiVertexBarrelHits/SiVertexBarrelHits.position.y']) for array in arrays])
    z = ak.concatenate([ak.flatten(array['SiVertexBarrelHits/SiVertexBarrelHits.position.z']) for array in arrays])
    
    # Calculate r and phi
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    # Count hits per cell
    hit_counts = Counter(cellids_combined.tolist())
    
    # Get geometry information if file provided
    geometry_info = None
    if geometry_file:
        geometry_info = get_geometry_info(geometry_file)
    
    # Analyze for each threshold
    threshold_stats = {}
    for threshold in hit_thresholds:
        # Decode all cellIDs and organize by layer
        layer_hits = {}
        for cellid, count in hit_counts.items():
            decoded = decode_cellID(cellid)
            layer = decoded['layer'] + 1  # Add 1 to match geometry
            if layer not in layer_hits:
                layer_hits[layer] = {'hit_counts': [], 'total_hits': 0, 'positions': {'r': [], 'phi': [], 'z': []}}
            layer_hits[layer]['hit_counts'].append(count)
            layer_hits[layer]['total_hits'] += count
        
        # Calculate statistics for this threshold
        threshold_stats[threshold] = {
            'total_cells_hit': len(hit_counts),
            'cells_above_threshold': sum(1 for count in hit_counts.values() if count >= threshold),
            'max_hits_per_cell': max(hit_counts.values()) if hit_counts else 0,
            'hit_distribution': dict(Counter(hit_counts.values())),
            'per_layer': {
                layer: {
                    'cells_hit': len(hits['hit_counts']),
                    'cells_above_threshold': sum(1 for count in hits['hit_counts'] if count >= threshold),
                    'total_hits': hits['total_hits'],
                    'max_hits': max(hits['hit_counts']),
                    'mean_hits': np.mean(hits['hit_counts']),
                    'occupancy': None  # Will fill if geometry info available
                }
                for layer, hits in layer_hits.items()
            }
        }
        
        # Add occupancy if geometry information available
        if geometry_info:
            for layer in threshold_stats[threshold]['per_layer']:
                threshold_stats[threshold]['per_layer'][layer]['occupancy'] = 0.0
                if layer in geometry_info['layers']:
                    total_cells = geometry_info['layers'][layer]['total_cells']
                    if total_cells > 0:
                        cells_above = threshold_stats[threshold]['per_layer'][layer]['cells_above_threshold']
                        threshold_stats[threshold]['per_layer'][layer]['occupancy'] = cells_above / total_cells
    
    return {
        'threshold_stats': threshold_stats,
        'positions': {
            'r': r,
            'phi': phi,
            'z': z
        }
    }


def summarize_stats_over_detector(stats, geometry_info, threshold=1):
    """
    Summarize total hits, total cells, and mean occupancy for a detector.

    - total hits: sum of per-layer total hits at the given threshold
    - total cells: sum of total cells across all layers
    - mean occupancy: total_hits / total_cells
    """
    # Sum total cells from geometry info
    total_cells = 0
    try:
        for layer_info in geometry_info.get('layers', {}).values():
            total_cells += int(layer_info.get('total_cells', 0) or 0)
    except Exception:
        total_cells = 0

    # Sum total hits from stats per-layer
    total_hits = 0
    try:
        per_layer = stats.get('threshold_stats', {}).get(threshold, {}).get('per_layer', {})
        for layer_stats in per_layer.values():
            total_hits += float(layer_stats.get('total_hits', 0) or 0)
    except Exception:
        total_hits = 0.0

    mean_occupancy = (total_hits / total_cells) if total_cells > 0 else 0.0

    return {
        'detector_name': geometry_info.get('detector_name', stats.get('detector_name', 'unknown')),
        'total_hits': total_hits,
        'total_cells': total_cells,
        'mean_occupancy': mean_occupancy,
    }

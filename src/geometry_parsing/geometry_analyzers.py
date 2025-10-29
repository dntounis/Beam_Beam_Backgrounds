import numpy as np 
from src.geometry_parsing.cellid_decoders import decode_cellid_tracker,create_decoder,decode_dd4hep_cellid


def analyze_vertex_barrel_layers(cellids, x_positions, y_positions):
    """
    Analyze SiVertexBarrel hits by comparing physical radius with layer ID
    
    Layer radii from XML:
    Layer 1: rc=15.05mm  (13.0-17.0mm)
    Layer 2: rc=23.03mm  (21.0-25.0mm)
    Layer 3: rc=35.79mm  (34.0-38.0mm)
    Layer 4: rc=47.50mm  (46.6-50.6mm)
    Layer 5: rc=59.90mm  (59.0-63.0mm)
    """
    
    # Define layer radii ranges (mm) based on envelope dimensions
    LAYER_RANGES = [
        (13.0, 17.0, 1),  # Layer 1
        (21.0, 25.0, 2),  # Layer 2 
        (34.0, 38.0, 3),  # Layer 3
        (46.6, 50.6, 4),  # Layer 4
        (59.0, 63.0, 5)   # Layer 5
    ]
    
    def get_layer_from_radius(r):
        """Determine layer from radius"""
        for rmin, rmax, layer in LAYER_RANGES:
            if rmin <= r <= rmax:
                return layer
        return None
    
    results = []
    mismatches = []
    
    # Calculate radii from x,y positions
    radii = np.sqrt(x_positions**2 + y_positions**2)
    
    for i, (cellid, x, y, r) in enumerate(zip(cellids, x_positions, y_positions, radii)):
        try:
            # Get layer from physical radius
            physical_layer = get_layer_from_radius(r)
            
            # Get layer from cellID using new decoder
            decoded = decode_cellid_tracker(cellid)
            cellid_layer = decoded['layer']
            
            result = {
                'cellid': cellid,
                'binary': format(int(cellid), '032b'),
                'x': x,
                'y': y,
                'r': r,
                'physical_layer': physical_layer,
                'cellid_layer': cellid_layer,
                'decoded': decoded
            }
            
            results.append(result)
            
            # Track mismatches between physical and cellID layers
            if physical_layer != cellid_layer:
                mismatches.append(result)
                
        except Exception as e:
            print(f"Error analyzing hit {i}, cellID {cellid}: {e}")
            continue
    
    # Print analysis
    print("\nSiVertexBarrel Layer Analysis")
    print("-" * 50)
    
    # Summary by physical radius
    print("\nHits by physical radius:")
    layer_hits = {layer: 0 for _, _, layer in LAYER_RANGES}
    no_layer = 0
    
    for r in results:
        if r['physical_layer'] is not None:
            layer_hits[r['physical_layer']] += 1
        else:
            no_layer += 1
    
    for layer in sorted(layer_hits):
        rmin, rmax, _ = next(r for r in LAYER_RANGES if r[2] == layer)
        print(f"Layer {layer} ({rmin:.1f}-{rmax:.1f}mm): {layer_hits[layer]} hits")
    if no_layer > 0:
        print(f"Outside layer ranges: {no_layer} hits")
    
    # Summary by cellID layer
    print("\nHits by cellID layer:")
    layer_counts = {}
    for r in results:
        layer = r['cellid_layer']
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    for layer in sorted(layer_counts):
        print(f"Layer {layer}: {layer_counts[layer]} hits")
    
    # Report mismatches
    print(f"\nFound {len(mismatches)} layer mismatches:")
    if mismatches:
        print("\nSample mismatches:")
        for m in mismatches[:10]:  # Show first 10
            print(f"\nCellID: {m['cellid']}")
            print(f"Binary: {m['binary']}")
            print(f"Position: x={m['x']:.1f}, y={m['y']:.1f}, r={m['r']:.1f}mm")
            print(f"Physical layer: {m['physical_layer']}")
            print(f"CellID layer: {m['cellid_layer']}")
            print(f"Decoded: {m['decoded']}")
            
            # Show bit groups for readability
            bit_groups = ' '.join([m['binary'][i:i+4] for i in range(0, len(m['binary']), 4)])
            print(f"Bit groups: {bit_groups}")
    
    return results, mismatches

def run_analysis(cellids, x_positions, y_positions):
    """Run the analysis and output results"""
    results, mismatches = analyze_vertex_barrel_layers(cellids, x_positions, y_positions)
    
    # Additional statistics
    layer_matching = len(results) - len(mismatches)
    match_percent = (layer_matching / len(results)) * 100 if results else 0
    
    print(f"\nAnalysis Summary")
    print("-" * 50)
    print(f"Total hits analyzed: {len(results)}")
    print(f"Layer matches: {layer_matching} ({match_percent:.1f}%)")
    print(f"Layer mismatches: {len(mismatches)}")
    
    return results, mismatches


def analyze_vertex_modules(cellids, x_positions, y_positions, detector="SiVertexBarrel"):
    """
    Analyze module distribution using proper DD4hep bit field decoding
    """
    # Layer definitions from XML
    LAYER_RANGES = [
        (13.0, 17.0, 1, 12),  # Layer 1: 12 modules
        (21.0, 25.0, 2, 12),  # Layer 2: 12 modules
        (34.0, 38.0, 3, 18),  # Layer 3: 18 modules
        (46.6, 50.6, 4, 24),  # Layer 4: 24 modules
        (59.0, 63.0, 5, 30)   # Layer 5: 30 modules
    ]
    
    def get_layer_from_radius(r):
        for rmin, rmax, layer, _ in LAYER_RANGES:
            if rmin <= r <= rmax:
                return layer
        return None
    
    # Convert inputs to numpy arrays if needed
    if not isinstance(x_positions, np.ndarray):
        x_positions = np.array(x_positions)
    if not isinstance(y_positions, np.ndarray):
        y_positions = np.array(y_positions)
    
    # Create decoder
    decoder = create_decoder(detector)
    
    # Calculate radii
    radii = np.sqrt(x_positions**2 + y_positions**2)
    
    # Store results
    decoded_values = []
    layer_matches = 0
    total_hits = 0
    
    # Analyze each hit
    for cellid, r in zip(cellids, radii):
        try:
            decoded = decoder.decode(cellid)
            decoded_values.append(decoded)
            
            physical_layer = get_layer_from_radius(r)
            if physical_layer is not None:
                total_hits += 1
                if physical_layer == decoded['layer']:
                    layer_matches += 1
                    
        except Exception as e:
            print(f"Error analyzing cellID {cellid}: {e}")
            continue
    
    # Print analysis
    print("\nSiVertexBarrel Analysis with DD4hep Decoder")
    print("-" * 50)
    
    print("\nLayer distribution by radius:")
    for rmin, rmax, layer, n_mod in LAYER_RANGES:
        count = sum(1 for r in radii if rmin <= r <= rmax)
        print(f"Layer {layer} ({rmin:.1f}-{rmax:.1f}mm): {count} hits (expect {n_mod} modules)")
    
    print("\nDecoded layer distribution:")
    layer_counts = {}
    for d in decoded_values:
        layer = d['layer']
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    for layer in sorted(layer_counts):
        print(f"Layer {layer}: {layer_counts[layer]} hits")
    
    # Layer matching statistics
    match_rate = (layer_matches / total_hits * 100) if total_hits > 0 else 0
    print(f"\nLayer matching: {layer_matches}/{total_hits} ({match_rate:.1f}%)")
    
    # Sample decoding
    print("\nSample decoded values:")
    for cellid, r in zip(cellids[:5], radii[:5]):
        decoded = decoder.decode(cellid)
        physical_layer = get_layer_from_radius(r)
        print(f"\nCellID: {cellid}")
        print(f"Binary: {format(int(cellid), '032b')}")
        print(f"Decoded: {decoded}")
        print(f"Radius: {r:.1f}mm")
        print(f"Physical layer: {physical_layer}")
    
    return decoded_values

def run_module_analysis_v2(cellids, x_positions, y_positions):
    """Run the analysis with proper DD4hep decoding"""
    return analyze_vertex_modules(cellids, x_positions, y_positions)



def analyze_cellid_patterns(cellids, detector_name):
    """
    Analyze and display cellID bit patterns for different detector types.
    
    Args:
        cellids: List of cellID integers
        detector_name: Name of the detector
        
    Returns:
        Dictionary of layer patterns with analysis
    """
    # Group detectors by readout scheme
    tracker_detectors = {
        "SiVertexBarrel", "SiVertexEndcap", 
        "SiTrackerBarrel", "SiTrackerEndcap", "SiTrackerForward"
    }
    
    calorimeter_detectors = {
        "ECalBarrel", "ECalEndcap",
        "HCalBarrel", "HCalEndcap", 
        "MuonBarrel", "MuonEndcap"
    }
    
    forward_calo_detectors = {
        "BeamCal", "LumiCal"
    }
    
    layer_patterns = {}
    
    for cellid in cellids:
        try:
            decoded = decode_dd4hep_cellid(cellid, detector_name)
            binary = format(cellid, '064b' if cellid > 0xFFFFFFFF else '032b')
            
            # Split binary into 4-bit groups for readability
            bit_groups = ' '.join(binary[i:i+4] for i in range(0, len(binary), 4))
            
            layer_key = decoded['layer']
            
            if layer_key not in layer_patterns:
                layer_patterns[layer_key] = {
                    'count': 0,
                    'sample_binary': binary,
                    'sample_decoded': decoded,
                    'bit_groups': bit_groups,
                    'unique_values': set()
                }
            
            # Create unique identifier based on detector type
            if detector_name in tracker_detectors:
                unique_id = f"m:{decoded['module']}_s:{decoded['sensor']}"
            elif detector_name in calorimeter_detectors:
                unique_id = f"m:{decoded['module']}_st:{decoded['stave']}_sm:{decoded['submodule']}_x:{decoded['x']}_y:{decoded['y']}"
            elif detector_name in forward_calo_detectors:
                unique_id = f"b:{decoded['barrel']}_sl:{decoded['slice']}_x:{decoded['x']}_y:{decoded['y']}"
            else:
                unique_id = str(cellid)
                
            layer_patterns[layer_key]['count'] += 1
            layer_patterns[layer_key]['unique_values'].add(unique_id)
            
        except Exception as e:
            print(f"Error analyzing cellID {cellid}: {e}")
            continue
    
    return layer_patterns

def print_layer_patterns(patterns, include_unique_values=False):
    """
    Print analyzed layer patterns in a readable format.
    
    Args:
        patterns: Dictionary returned by analyze_cellid_patterns
        include_unique_values: Whether to print the set of unique value combinations
    """
    print("\nLayer patterns found:")
    print("-" * 50 + "\n")
    
    for layer in sorted(patterns.keys()):
        pattern = patterns[layer]
        print(f"Layer {layer}:")
        print(f"  Number of cells: {pattern['count']}")
        print(f"  Sample binary:   {pattern['sample_binary']}")
        print(f"  Sample decoded:   {pattern['sample_decoded']}")
        print(f"  Bit groups:      {pattern['bit_groups']}")
        # if include_unique_values:
        #     print("\n  Unique value combinations:")
        #     for val in sorted(pattern['unique_values']):
        #         print(f"    {val}")
        print()


def analyze_layer_hits(hit_counts, layer_number):
    """
    Analyze hits for a specific layer
    
    Parameters:
    -----------
    hit_counts : dict
        Dictionary of cellID -> hit count
    layer_number : int
        Layer to analyze
        
    Returns:
    --------
    dict with layer hit statistics
    """
    # Only consider cells from this layer
    layer_cells = {
        cellid: count for cellid, count in hit_counts.items()
        if decode_cellID(cellid)['layer'] + 1 == layer_number
    }
    
    if not layer_cells:
        return {
            'cells_hit': 0,
            'total_hits': 0,
            'hit_counts': [],
            'max_hits': 0,
            'mean_hits': 0
        }
    
    return {
        'cells_hit': len(layer_cells),
        'total_hits': sum(layer_cells.values()),
        'hit_counts': list(layer_cells.values()),
        'max_hits': max(layer_cells.values()),
        'mean_hits': np.mean(list(layer_cells.values()))
    }



def analyze_layer_distribution(hit_counts):
    """
    Analyze the distribution of layers in the hits
    """
    layer_counts = {}
    raw_layers = {}
    
    for cellid in hit_counts.keys():
        decoded = decode_cellID(cellid)
        raw_layer = decoded['layer']  # Before adding 1
        adjusted_layer = raw_layer + 1  # After adding 1
        
        if raw_layer not in raw_layers:
            raw_layers[raw_layer] = 0
        raw_layers[raw_layer] += 1
        
        if adjusted_layer not in layer_counts:
            layer_counts[adjusted_layer] = 0
        layer_counts[adjusted_layer] += 1
    
    print("\nLayer Distribution Analysis:")
    print("-" * 50)
    print("Raw layer values (before +1 adjustment):")
    for layer, count in sorted(raw_layers.items()):
        print(f"  Layer {layer}: {count} cells")
    
    print("\nAdjusted layer values:")
    for layer, count in sorted(layer_counts.items()):
        print(f"  Layer {layer}: {count} cells")
    
    # Sample of cellIDs and their decoded values
    print("\nSample cellID decoding:")
    print("-" * 50)
    sample_size = min(5, len(hit_counts))
    sample_cellids = list(hit_counts.keys())[:sample_size]
    
    for cellid in sample_cellids:
        decoded = decode_cellID(cellid)
        print(f"CellID: {cellid}")
        print(f"  Binary: {format(cellid, '032b')}")
        print(f"  Decoded: {decoded}")
        print()



def print_layer_info(layer_info, detector_type):
    """Print layer information based on detector type"""
    if 'rings' in layer_info:  # Tracker endcap with rings
        print(f"    total_cells: {layer_info['total_cells']}")
        print("    Rings:")
        for i, ring in enumerate(layer_info['rings'], 1):
            print(f"      Ring {i}:")
            for key, value in sorted(ring.items()):
                if key not in ['cells_per_module', 'module_type']:
                    print(f"        {key}: {value}")
    else:  # Regular layer
        for key, value in sorted(layer_info.items()):
            if key not in ['cells_per_module', 'module_type']:
                print(f"    {key}: {value}")

import os
import re
from typing import Dict, List, Tuple

import xml.etree.ElementTree as ET
import numpy as np 
import math 


# Allow overriding the assumed radial orientation of trapezoid modules via env var.
# Format: "DetectorName:radial_x,OtherDetector:radial_z"
def _build_trd_orientation_overrides() -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    env_value = os.getenv("DD4HEP_TRD_RADIAL_AXIS", "")
    if not env_value:
        return overrides
    for item in env_value.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        det_name, axis = item.split(":", 1)
        axis = axis.strip().lower()
        if axis not in {"radial_x", "radial_z"}:
            continue
        overrides[det_name.strip().lower()] = axis
    return overrides


_TRD_ORIENTATION_OVERRIDES = _build_trd_orientation_overrides()
DEFAULT_TRD_ORIENTATION = "radial_x"



def evaluate_constant_expression(expr_str, constants):
    """
    Evaluate an expression that may contain constant references and basic math.
    
    Parameters:
    -----------
    expr_str : str
        Expression to evaluate (e.g., "ECalBarrel_rmin - 10*env_safety")
    constants : dict
        Dictionary of constants
        
    Returns:
    --------
    float or None
    """
    if not isinstance(expr_str, str):
        return expr_str
        
    try:
        # First substitute all constants
        expr = expr_str
        for name, value in constants.items():
            if isinstance(value, (int, float)):
                expr = expr.replace(name, str(value))
        
        # Now evaluate the expression
        expr = expr.replace('*', ' * ').replace('/', ' / ').replace('+', ' + ').replace('-', ' - ')
        terms = expr.split()
        result = 0
        operator = '+'
        
        for term in terms:
            if term in ['+', '-', '*', '/']:
                operator = term
                continue
                
            try:
                value = float(term)
                if operator == '+':
                    result += value
                elif operator == '-':
                    result -= value
                elif operator == '*':
                    result *= value
                elif operator == '/':
                    result /= value
            except ValueError:
                return None
                
        return result
    except:
        return None

def parse_repeat_value(layer_elem, constants):
    """
    Parse repeat attribute with better error handling.
    
    Parameters:
    -----------
    layer_elem : xml.etree.ElementTree.Element
        Layer element
    constants : dict
        Constants dictionary
        
    Returns:
    --------
    int : repeat count
    """
    repeat_str = layer_elem.get('repeat', '1')
    
    try:
        # First try direct integer conversion
        return int(repeat_str)
    except ValueError:
        # Try evaluating as expression
        value = evaluate_constant_expression(repeat_str, constants)
        if value is not None:
            return int(value)
        return 1


def _get_trd_orientation(config) -> str:
    """
    Determine how to interpret trapezoid parameters for the provided detector.
    Returns either 'radial_x' (historical behaviour) or 'radial_z'.
    
    For SiD disk-type detectors (vertex/tracker endcap/forward), the k4geo drivers
    apply rotation RotationZYX(0, -π/2-φ, -π/2) which maps:
    - TRD local z → global radial direction
    - TRD local x1/x2 → global azimuthal widths
    
    Therefore for these detectors we need 'radial_z' orientation.
    """
    name = getattr(config, "name", "").strip().lower()
    if name in _TRD_ORIENTATION_OVERRIDES:
        return _TRD_ORIENTATION_OVERRIDES[name]

    detector_class = getattr(config, "detector_class", "").lower()
    detector_type = getattr(config, "detector_type", "").lower()

    # SiD disk-type detectors use k4geo drivers that rotate TRD so z becomes radial
    if detector_class in {"vertex", "tracker"} and detector_type in {"endcap", "forward"}:
        return "radial_z"

    return DEFAULT_TRD_ORIENTATION

def handle_layer_id(layer_elem, repeat_index=0):
    """
    Get layer ID with better handling of repeat groups.
    
    Parameters:
    -----------
    layer_elem : xml.etree.ElementTree.Element
        Layer element
    repeat_index : int
        Index within repeat group
        
    Returns:
    --------
    int or None : layer ID
    """
    # First try direct id attribute
    layer_id_str = layer_elem.get('id')
    base_id = None
    
    if layer_id_str is not None:
        try:
            base_id = int(layer_id_str)
        except ValueError:
            pass
    
    # If no direct ID, try to generate one based on position
    if base_id is None:
        # Find position among siblings
        parent = layer_elem.getparent()
        if parent is not None:
            siblings = parent.findall('layer')
            try:
                base_id = siblings.index(layer_elem) + 1
            except ValueError:
                return None
    
    if base_id is not None:
        return base_id + repeat_index
    
    return None

def parse_layer_sensitive_slices(layer_elem):
    """
    Count number of sensitive slices in a layer.
    
    Parameters:
    -----------
    layer_elem : xml.etree.ElementTree.Element
        Layer element
        
    Returns:
    --------
    int : number of sensitive slices
    """
    sensitive_slices = layer_elem.findall(".//slice[@sensitive='yes']")
    return len(sensitive_slices)


# def parse_calorimeter_geometry(detector, config, geometry_info, constants):
#     """Parse calorimeter-type detector geometry"""
#     # Get detector dimensions
#     prefix = config.name
    
#     rmin = constants.get(f'{prefix}_rmin')
#     rmax = constants.get(f'{prefix}_rmax')
    
#     if prefix.endswith('Barrel'):
#         half_length = constants.get(f'{prefix}_half_length')
#         z_length = 2 * half_length if half_length else None
#     else:
#         zmin = constants.get(f'{prefix}_zmin')
#         zmax = constants.get(f'{prefix}_zmax')
#         z_length = zmax - zmin if zmin and zmax else None
    
#     # Process layers
#     layer_count = 0
#     for layer_elem in detector.findall('.//layer'):
#         # Handle repeats and layer numbering
#         repeat = parse_repeat_value(layer_elem, constants)
#         n_sensitive = parse_layer_sensitive_slices(layer_elem)
        
#         # Get cell size
#         cell_size = config.get_cell_size()
        
#         # Calculate cells per layer base
#         if rmin and rmax and z_length:
#             if prefix.endswith('Barrel'):
#                 # For barrel: circumference * length
#                 circumference = 2 * np.pi * ((rmax + rmin) / 2)
#                 cells_in_phi = int(circumference / cell_size['x'])
#                 cells_in_z = int(z_length / cell_size['y'])
#                 cells_per_layer = cells_in_phi * cells_in_z
#             else:
#                 # For endcap: area of disk
#                 area = np.pi * (rmax**2 - rmin**2)
#                 cells_per_layer = int(area / (cell_size['x'] * cell_size['y']))
            
#             # Process each repeat
#             for r in range(repeat):
#                 layer_id = handle_layer_id(layer_elem, r)
#                 if layer_id is not None:
#                     total_cells = cells_per_layer * n_sensitive
                    
#                     layer_info = {
#                         'rmin': rmin,
#                         'rmax': rmax,
#                         'cells_per_layer': cells_per_layer,
#                         'sensitive_slices': n_sensitive,
#                         'total_cells': total_cells,
#                         'repeat_group': layer_count,
#                         'repeat_number': r
#                     }
                    
#                     if prefix.endswith('Barrel'):
#                         layer_info['z_length'] = z_length
#                     else:
#                         layer_info['zmin'] = zmin
#                         layer_info['zmax'] = zmax
                    
#                     geometry_info['layers'][layer_id] = layer_info
#                     geometry_info['total_cells'] += total_cells
                    
#                     layer_count += 1


def parse_calorimeter_geometry(detector, config, geometry_info, constants):
    """Parse calorimeter-type detector geometry"""
    # Get detector dimensions
    prefix = config.name
    
    # Get basic dimensions
    rmin = constants.get(f'{prefix}_rmin')
    rmax = constants.get(f'{prefix}_rmax')
    
    if prefix.endswith('Barrel'):
        half_length = constants.get(f'{prefix}_half_length')
        z_length = 2 * half_length if half_length else None
        geometry_info.update({
            'rmin': rmin,
            'rmax': rmax,
            'z_length': z_length
        })
    else:
        zmin = constants.get(f'{prefix}_zmin')
        zmax = constants.get(f'{prefix}_zmax')
        geometry_info.update({
            'rmin': rmin,
            'rmax': rmax,
            'zmin': zmin,
            'zmax': zmax
        })
    
    # Process layers
    layer_count = 0
    for layer_elem in detector.findall('.//layer'):
        # Get layer info with proper repeat handling
        layer_info = parse_calorimeter_layer(layer_elem, constants)
        if not layer_info:
            continue
        
        # Handle repeats
        repeat = layer_info['repeat']
        
        # Calculate cells for each repeat
        if config.detector_type == 'barrel':
            # For barrel: circumference * length
            circumference = 2 * np.pi * ((rmax + rmin) / 2)
            cell_size = config.get_cell_size()
            cells_in_phi = int(circumference / cell_size['x'])
            cells_in_z = int(z_length / cell_size['y'])
            cells_per_layer = cells_in_phi * cells_in_z
        else:
            # For endcap: area of disk
            area = np.pi * (rmax**2 - rmin**2)
            cell_size = config.get_cell_size()
            cells_per_layer = int(area / (cell_size['x'] * cell_size['y']))
        
        # Store info for each repeat
        for r in range(repeat):
            layer_id = layer_count + r
            
            # Get number of sensitive slices
            n_sensitive = len(layer_info['sensitive_slices'])
            total_cells = cells_per_layer * n_sensitive
            
            # Store layer info
            layer_data = {
                'repeat_group': layer_count,
                'repeat_number': r,
                'sensitive_slices': layer_info['sensitive_slices'],
                'cells_per_layer': cells_per_layer,
                'total_cells': total_cells,
                **{k:v for k,v in layer_info.items() if k not in ['repeat', 'sensitive_slices']}
            }
            
            # Add envelope bounds from detector-level or layer-level
            if config.detector_type == 'barrel':
                # Copy detector-level bounds to layer
                if 'rmin' in geometry_info:
                    layer_data['inner_r'] = geometry_info['rmin']
                if 'rmax' in geometry_info:
                    layer_data['outer_r'] = geometry_info['rmax']
                if 'z_length' in geometry_info:
                    layer_data['z_length'] = geometry_info['z_length']
                # Also check layer_info for explicit bounds
                if 'rmin' in layer_info:
                    layer_data['inner_r'] = layer_info['rmin']
                if 'rmax' in layer_info:
                    layer_data['outer_r'] = layer_info['rmax']
            else:
                # Endcap detectors
                if 'rmin' in geometry_info:
                    layer_data['inner_r'] = geometry_info['rmin']
                if 'rmax' in geometry_info:
                    layer_data['outer_r'] = geometry_info['rmax']
                if 'zmin' in geometry_info:
                    layer_data['z_min'] = geometry_info['zmin']
                if 'zmax' in geometry_info:
                    layer_data['z_max'] = geometry_info['zmax']
                # Also check layer_info
                if 'rmin' in layer_info:
                    layer_data['inner_r'] = layer_info['rmin']
                if 'rmax' in layer_info:
                    layer_data['outer_r'] = layer_info['rmax']
                if 'zmin' in layer_info:
                    layer_data['z_min'] = layer_info['zmin']
                if 'zmax' in layer_info:
                    layer_data['z_max'] = layer_info['zmax']
            
            geometry_info['layers'][layer_id] = layer_data
            
            # Add to total
            geometry_info['total_cells'] += total_cells
        
        layer_count += repeat

def parse_calorimeter_layer(layer_elem, constants):
    """
    Parse a calorimeter/muon detector layer, handling repeats and slices.
    
    Parameters:
    -----------
    layer_elem : xml.etree.ElementTree.Element
        Layer XML element 
    constants : dict
        Constants dictionary
        
    Returns:
    --------
    dict with layer information
    """
    layer_info = {}
    
    # Get repeat count
    repeat = parse_repeat_value(layer_elem, constants)
    layer_info['repeat'] = repeat
    
    # Count sensitive slices
    sensitive_slices = []
    for i, slice_elem in enumerate(layer_elem.findall('slice')):
        if slice_elem.get('sensitive', '').lower() == 'yes':
            sensitive_slices.append(i)
    layer_info['sensitive_slices'] = sensitive_slices
    
    # Get dimensions if available
    for dim in ['rmin', 'rmax', 'zmin', 'zmax']:
        value = parse_value(layer_elem.get(dim), constants)
        if value is not None:
            layer_info[dim] = value
    
    # Get segmentation info
    seg_elem = layer_elem.find('.//segmentation')
    if seg_elem is not None:
        grid_x = parse_value(seg_elem.get('grid_size_x'), constants)
        grid_y = parse_value(seg_elem.get('grid_size_y'), constants)
        if grid_x is not None and grid_y is not None:
            layer_info['cell_size'] = {'x': grid_x, 'y': grid_y}
    
    return layer_info

def parse_forward_calo_layer(layer_elem, constants):
    """
    Parse a forward calorimeter layer with slices.
    
    Parameters:
    -----------
    layer_elem : xml.etree.ElementTree.Element
        Layer XML element
    constants : dict
        Constants dictionary
        
    Returns:
    --------
    dict with layer information
    """
    layer_info = {}
    
    # Get repeat count
    repeat = parse_repeat_value(layer_elem, constants)
    layer_info['repeat'] = repeat
    
    # Count sensitive slices
    sensitive_slices = []
    total_thickness = 0
    
    for i, slice_elem in enumerate(layer_elem.findall('slice')):
        thickness = parse_value(slice_elem.get('thickness'), constants)
        if thickness is not None:
            total_thickness += thickness
            
        if slice_elem.get('sensitive', '').lower() == 'yes':
            sensitive_slices.append({
                'index': i,
                'thickness': thickness,
                'z_offset': total_thickness - thickness/2  # Center of slice
            })
    
    layer_info.update({
        'sensitive_slices': sensitive_slices,
        'total_thickness': total_thickness
    })
    
    return layer_info

def parse_forward_calo_geometry(detector, config, geometry_info, constants):
    """
    Parse forward calorimeter geometry (BeamCal, LumiCal).
    
    Parameters as before.
    """
    # Get detector dimensions
    dims = detector.find('.//dimensions')
    if dims is not None:
        geometry_info.update({
            'inner_r': parse_value(dims.get('inner_r'), constants),
            'outer_r': parse_value(dims.get('outer_r'), constants),
            'inner_z': parse_value(dims.get('inner_z'), constants)
        })
    
    # Get segmentation info
    readout = detector.find('.//readout')
    if readout is not None:
        seg = readout.find('.//segmentation')
        if seg is not None:
            geometry_info['grid_size_x'] = parse_value(seg.get('grid_size_x'), constants)
            geometry_info['grid_size_y'] = parse_value(seg.get('grid_size_y'), constants)
    
    # Process layers
    layer_count = 0
    z_pos = geometry_info.get('inner_z', 0)
    
    for layer_elem in detector.findall('.//layer'):
        # Get layer info
        layer_info = parse_forward_calo_layer(layer_elem, constants)
        if not layer_info:
            continue
            
        # Handle repeats
        repeat = layer_info['repeat']
        
        # Calculate cells per layer
        r_range = geometry_info['outer_r'] - geometry_info['inner_r']
        cell_size = config.get_cell_size()
        grid_size = int(r_range / cell_size['x'])
        cells_per_layer = grid_size * grid_size  # Square grid
        
        # Store info for each repeat
        for r in range(repeat):
            layer_id = layer_count + r
            
            # Calculate total cells including sensitive slices
            n_sensitive = len(layer_info['sensitive_slices'])
            total_cells = cells_per_layer * n_sensitive
            
            # Store layer info
            geometry_info['layers'][layer_id] = {
                'zstart': z_pos,
                'repeat_group': layer_count,
                'repeat_number': r,
                'sensitive_slices': layer_info['sensitive_slices'],
                'cells_per_layer': cells_per_layer,
                'total_cells': total_cells,
                'inner_r': geometry_info['inner_r'],
                'outer_r': geometry_info['outer_r'],
                'grid_size': grid_size
            }
            
            # Add to total
            geometry_info['total_cells'] += total_cells
            
            # Update z position
            z_pos += layer_info['total_thickness']
            
        layer_count += repeat


def extract_barrel_layer_info(layer_elem, constants=None):
    """
    Extract geometry information for a barrel detector layer.
    
    Parameters:
    -----------
    layer_elem : xml.etree.ElementTree.Element
        XML element containing the layer information
    constants : dict, optional
        Dictionary of constants for value parsing
        
    Returns:
    --------
    dict containing layer geometry information
    """
    info = {}
    
    # Get layer ID
    layer_id = int(layer_elem.get('id', 0))
    info['id'] = layer_id
    
    # Get barrel envelope info
    barrel_env = layer_elem.find('barrel_envelope')
    if barrel_env is not None:
        info['inner_r'] = parse_value(barrel_env.get('inner_r'), constants)
        info['outer_r'] = parse_value(barrel_env.get('outer_r'), constants)
        info['z_length'] = parse_value(barrel_env.get('z_length'), constants)
    
    # Get rphi layout info
    rphi_elem = layer_elem.find('rphi_layout')
    if rphi_elem is not None:
        info['phi_tilt'] = parse_value(rphi_elem.get('phi_tilt'), constants)
        info['nphi'] = int(rphi_elem.get('nphi', 0))
        info['phi0'] = parse_value(rphi_elem.get('phi0'), constants)
        info['rc'] = parse_value(rphi_elem.get('rc'), constants)
        info['dr'] = parse_value(rphi_elem.get('dr'), constants)
    
    # Get z layout info
    z_elem = layer_elem.find('z_layout')
    if z_elem is not None:
        info['nz'] = int(z_elem.get('nz', 0))
        info['z0'] = parse_value(z_elem.get('z0'), constants)
        info['dr'] = parse_value(z_elem.get('dr'), constants)
    
    return info

def extract_endcap_ring_info(ring_elem, constants=None):
    """
    Extract geometry information for an endcap detector ring.
    
    Parameters:
    -----------
    ring_elem : xml.etree.ElementTree.Element
        XML element containing the ring information
    constants : dict, optional
        Dictionary of constants for value parsing
        
    Returns:
    --------
    dict containing ring geometry information
    """
    info = {}
    
    # Basic ring parameters
    info['inner_r'] = parse_value(ring_elem.get('r'), constants)
    info['zstart'] = parse_value(ring_elem.get('zstart'), constants)
    info['nmodules'] = int(ring_elem.get('nmodules', 0))
    info['dz'] = parse_value(ring_elem.get('dz'), constants)
    
    # Optional phi0 offset
    phi0 = ring_elem.get('phi0')
    if phi0:
        info['phi0'] = parse_value(phi0, constants)
    
    return info

def extract_module_dimensions(detector_elem):
    """
    Extract module dimensions from detector XML.
    
    Parameters:
    -----------
    detector_elem : xml.etree.ElementTree.Element
        XML element containing the detector information
        
    Returns:
    --------
    dict containing module dimensions
    """
    dims = {}
    
    # Look for module elements
    module_elems = detector_elem.findall(".//module")
    if module_elems:
        for module in module_elems:
            # Get module name
            module_name = module.get('name', '')
            
            # Look for trd or module_envelope element
            trd = module.find('trd')
            env = module.find('module_envelope')
            
            if trd is not None:
                dims[module_name] = {
                    'width1': parse_value(trd.get('x1')),
                    'width2': parse_value(trd.get('x2')),
                    'length': parse_value(trd.get('z'))
                }
            elif env is not None:
                dims[module_name] = {
                    'width': parse_value(env.get('width')),
                    'length': parse_value(env.get('length'))
                }
    
    return dims

def calculate_module_position(layer_info, module_idx, geometry_type='barrel'):
    """Updated module position calculation"""
    if geometry_type == 'barrel':
        # For barrel detector
        if 'rc' not in layer_info:
            # Try to calculate rc from inner/outer radii
            if 'inner_r' in layer_info and 'outer_r' in layer_info:
                rc = (layer_info['inner_r'] + layer_info['outer_r']) / 2
            else:
                return None  # Can't calculate position without radius info
        else:
            rc = layer_info['rc']
            
        nphi = layer_info.get('nphi', 1)
        phi_tilt = layer_info.get('phi_tilt', 0.0)
        phi0 = layer_info.get('phi0', 0.0)
        
        # Calculate module center in phi
        delta_phi = 2 * np.pi / nphi
        module_phi = phi0 + (module_idx + 0.5) * delta_phi + phi_tilt
        
        # Calculate cartesian coordinates
        x = rc * np.cos(module_phi)
        y = rc * np.sin(module_phi)
        z = 0.0
        
        return (x, y, z, module_phi)
        
    elif geometry_type == 'endcap':
        # For endcap detector
        if 'r' in layer_info:
            # Direct ring information
            r = layer_info['r']
            z = layer_info['zstart']
            nmodules = layer_info['nmodules']
        else:
            # Try to calculate from min/max
            if 'inner_r' in layer_info and 'outer_r' in layer_info:
                r = (layer_info['inner_r'] + layer_info['outer_r']) / 2
            else:
                return None
                
            if 'z' not in layer_info:
                return None
            z = layer_info['z']
            nmodules = layer_info.get('nmodules', 1)
        
        # Calculate position around the ring
        phi = (2 * np.pi / nmodules) * (module_idx + 0.5)
        if 'phi0' in layer_info:
            phi += layer_info['phi0']
        
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        
        return (x, y, z, phi)
    
    return None



def parse_detector_constants(main_xml_file, detector_name=None):
    """Parse detector constants with dependency resolution"""
    tree = ET.parse(main_xml_file)
    root = tree.getroot()
    
    debug = {}  # Store debug info per constant
    
    # First pass: collect all raw constant definitions
    raw_constants = {}
    for constant in root.findall('.//constant'):
        name = constant.get('name')
        value_str = constant.get('value')
        raw_constants[name] = value_str
        debug[name] = {
            'raw_value': value_str,
            'dependencies': set(),
            'evaluation_steps': []
        }
    
    # Add SiTrackerBarrel specific constants and dependencies
    if detector_name == 'SiTrackerBarrel':
        tracker_base = {
            'SiTracker_module_z_spacing': '42.2*mm',
            'SiTracker_circular_spacing': '12.2*mm',
            'SiTracker_tanTheta': '0.7',
            'SiTracker_bIntercept': '0.0*mm',
            'SiTrackerBarrel_module_width': '97.97*mm',
            'SiTrackerBarrel_module_length': '97.97*mm',
            'SiTrackerBarrel_inner_rc': '216.355*mm',
            'SiTrackerBarrel_outer_rc': '1216.355*mm',
            'SiTrackerBarrel_inner_z0': '422.7*mm',
            'SiTrackerBarrel_outer_z0': '1447.8*mm'
        }
        
        # Add tracker base constants if not already defined
        for name, value in tracker_base.items():
            if name not in raw_constants:
                raw_constants[name] = value
                debug[name] = {
                    'raw_value': value,
                    'dependencies': set(),
                    'evaluation_steps': []
                }
        
        # Add derived r spacing and rc values
        tracker_derived = {
            'SiTrackerBarrel_r_spacing': '(SiTrackerBarrel_outer_rc - SiTrackerBarrel_inner_rc)/4',
            'SiTrackerBarrel_rc_2': 'SiTrackerBarrel_inner_rc + SiTrackerBarrel_r_spacing',
            'SiTrackerBarrel_rc_3': 'SiTrackerBarrel_rc_2 + SiTrackerBarrel_r_spacing',
            'SiTrackerBarrel_rc_4': 'SiTrackerBarrel_rc_3 + SiTrackerBarrel_r_spacing',
            'SiTrackerBarrel_z0_2': '(SiTrackerBarrel_rc_2 - SiTracker_bIntercept)/SiTracker_tanTheta',
            'SiTrackerBarrel_z0_3': '(SiTrackerBarrel_rc_3 - SiTracker_bIntercept)/SiTracker_tanTheta',
            'SiTrackerBarrel_z0_4': '(SiTrackerBarrel_rc_4 - SiTracker_bIntercept)/SiTracker_tanTheta'
        }
        raw_constants.update(tracker_derived)
        for name, value in tracker_derived.items():
            debug[name] = {
                'raw_value': value,
                'dependencies': set(),
                'evaluation_steps': []
            }
    
    # Rest of the evaluation function remains the same...
    def evaluate_constant(name, visited=None):
        if visited is None:
            visited = set()
            
        if name in visited:
            if detector_name and detector_name in name:
                print(f"Warning: Circular dependency detected for {name}")
                print(f"  Dependency chain: {' -> '.join(visited)} -> {name}")
            return None
            
        visited.add(name)
        
        if name not in raw_constants:
            return None
            
        value_str = raw_constants[name]
        
        if isinstance(value_str, (int, float)):
            return value_str
            
        try:
            if detector_name and detector_name in name:
                debug[name]['evaluation_steps'].append(f"Starting with: {value_str}")
            
            while True:
                found_constant = False
                for ref_name in raw_constants:
                    if ref_name in value_str:
                        debug[name]['dependencies'].add(ref_name)
                        ref_value = evaluate_constant(ref_name, visited.copy())
                        if ref_value is not None:
                            value_str = value_str.replace(ref_name, str(ref_value))
                            found_constant = True
                            if detector_name and detector_name in name:
                                debug[name]['evaluation_steps'].append(
                                    f"After replacing {ref_name}: {value_str}")
                
                if not found_constant:
                    break
            
            result = parse_value(value_str, raw_constants)
            if result is not None:
                raw_constants[name] = result
                if detector_name and detector_name in name:
                    debug[name]['evaluation_steps'].append(f"Final result: {result}")
            return result
            
        except Exception as e:
            if detector_name and detector_name in name:
                print(f"Warning: Could not evaluate constant {name}: {str(e)}")
                print(f"  Current value string: {value_str}")
            return None

    # Add final module count calculations for tracker barrel
    if detector_name == 'SiTrackerBarrel':
        constants = {}
        for name in raw_constants:
            value = evaluate_constant(name)
            if value is not None:
                constants[name] = value

        inner_rc = constants.get('SiTrackerBarrel_inner_rc')
        outer_rc = constants.get('SiTrackerBarrel_outer_rc')
        if inner_rc is not None and outer_rc is not None:
            r_spacing = constants.get('SiTrackerBarrel_r_spacing')
            if r_spacing is None:
                r_spacing = (outer_rc - inner_rc) / 4.0
                constants['SiTrackerBarrel_r_spacing'] = r_spacing

            rc_progression = {
                '2': inner_rc + r_spacing,
                '3': inner_rc + 2.0 * r_spacing,
                '4': inner_rc + 3.0 * r_spacing,
            }
            for suffix, rc_val in rc_progression.items():
                constants.setdefault(f'SiTrackerBarrel_rc_{suffix}', rc_val)

            tan_theta = constants.get('SiTracker_tanTheta')
            b_intercept = constants.get('SiTracker_bIntercept', 0.0)
            if tan_theta:
                for suffix, rc_val in rc_progression.items():
                    z0_name = f'SiTrackerBarrel_z0_{suffix}'
                    constants.setdefault(z0_name, (rc_val - b_intercept) / tan_theta)

        circular_spacing = constants.get('SiTracker_circular_spacing')
        z_spacing = constants.get('SiTracker_module_z_spacing')
        layer_specs = [
            ('inner', 'SiTrackerBarrel_inner_rc', 'SiTrackerBarrel_inner_z0'),
            ('2', 'SiTrackerBarrel_rc_2', 'SiTrackerBarrel_z0_2'),
            ('3', 'SiTrackerBarrel_rc_3', 'SiTrackerBarrel_z0_3'),
            ('4', 'SiTrackerBarrel_rc_4', 'SiTrackerBarrel_z0_4'),
            ('outer', 'SiTrackerBarrel_outer_rc', 'SiTrackerBarrel_outer_z0'),
        ]

        for prefix, rc_name, z0_name in layer_specs:
            rc_val = constants.get(rc_name)
            if rc_val is None:
                continue

            if prefix in ['inner', 'outer']:
                nphi_name = f'SiTrackerBarrel_{prefix}_nphi'
                nz_name = f'SiTrackerBarrel_{prefix}_nz'
            else:
                nphi_name = f'SiTrackerBarrel_nphi_{prefix}'
                nz_name = f'SiTrackerBarrel_nz_{prefix}'

            if circular_spacing:
                constants.setdefault(nphi_name, math.floor(rc_val / circular_spacing))

            z0_val = constants.get(z0_name)
            if z0_val is not None and z_spacing:
                constants.setdefault(nz_name, 1 + math.floor(z0_val / z_spacing))

        # Ensure final values exist even if earlier substitutions failed
        for prefix, rc_name, z0_name in layer_specs:
            rc_val = constants.get(rc_name)
            z0_val = constants.get(z0_name)
            if rc_val is None or z0_val is None:
                continue

            if prefix in ['inner', 'outer']:
                nphi_name = f'SiTrackerBarrel_{prefix}_nphi'
                nz_name = f'SiTrackerBarrel_{prefix}_nz'
            else:
                nphi_name = f'SiTrackerBarrel_nphi_{prefix}'
                nz_name = f'SiTrackerBarrel_nz_{prefix}'

            if circular_spacing:
                constants[nphi_name] = math.floor(rc_val / circular_spacing)
            if z_spacing:
                constants[nz_name] = 1 + math.floor(z0_val / z_spacing)

        return constants

    # Evaluate all constants
    constants = {}
    for name in raw_constants:
        value = evaluate_constant(name)
        if value is not None:
            constants[name] = value
    
    return constants

def evaluate_expression(expr_str, constants):
    """Evaluate a mathematical expression with unit handling"""
    if not isinstance(expr_str, str):
        return expr_str
        
    # Remove units
    expr = expr_str.replace('*mm', '').replace('*cm', '').replace('*m', '')
    
    # Handle parentheses
    if '(' in expr and ')' in expr:
        # Extract content within parentheses
        start = expr.find('(')
        end = expr.find(')')
        inner_expr = expr[start+1:end]
        inner_value = evaluate_expression(inner_expr, constants)
        expr = expr[:start] + str(inner_value) + expr[end+1:]
    
    # Handle basic arithmetic
    try:
        if '+' in expr:
            parts = expr.split('+')
            return sum(evaluate_expression(p.strip(), constants) for p in parts)
        elif '-' in expr and not expr.startswith('-'):
            parts = expr.split('-')
            values = [evaluate_expression(p.strip(), constants) for p in parts]
            return values[0] - sum(values[1:])
        elif '*' in expr:
            parts = expr.split('*')
            result = 1
            for p in parts:
                result *= evaluate_expression(p.strip(), constants)
            return result
        elif '/' in expr:
            parts = expr.split('/')
            values = [evaluate_expression(p.strip(), constants) for p in parts]
            return values[0] / values[1]
        
        # Handle floor function
        if expr.startswith('floor(') and expr.endswith(')'):
            inner = expr[6:-1]
            value = evaluate_expression(inner, constants)
            return math.floor(value)
            
        # Try direct float conversion
        try:
            return float(expr)
        except ValueError:
            # Look up in constants
            if expr in constants:
                return constants[expr]
            return None
    except:
        return None


def evaluate_math_expression(expr_str):
    """
    Evaluate a mathematical expression string with support for parentheses and operator precedence.
    
    Parameters:
    -----------
    expr_str : str
        Expression to evaluate (e.g., "(1.75 + 787.105) * 2")
        
    Returns:
    --------
    float or None
    """
    if not isinstance(expr_str, str):
        return expr_str
        
    try:
        expr = expr_str.strip()
        
        # Handle parentheses recursively
        while '(' in expr:
            # Find innermost parentheses
            start = expr.rfind('(')
            if start == -1:
                break
            end = expr.find(')', start)
            if end == -1:
                break
            
            # Evaluate inner expression
            inner = expr[start+1:end]
            inner_value = evaluate_math_expression(inner)
            if inner_value is None:
                return None
            
            # Replace parentheses group with evaluated value
            expr = expr[:start] + str(inner_value) + expr[end+1:]
        
        # Remove outer parentheses if they wrap entire expression
        expr = expr.strip()
        while expr.startswith('(') and expr.endswith(')'):
            expr = expr[1:-1].strip()
        
        # Evaluate operations with proper precedence
        # First handle multiplication and division
        while '*' in expr or '/' in expr:
            # Find first * or / operation
            mul_idx = expr.find('*') if '*' in expr else len(expr)
            div_idx = expr.find('/') if '/' in expr else len(expr)
            
            if mul_idx < div_idx:
                # Handle multiplication
                left = expr[:mul_idx].strip()
                right = expr[mul_idx+1:].strip()
                # Find boundaries for left and right operands
                left_start = 0
                for i in range(mul_idx - 1, -1, -1):
                    if expr[i] in '+-' and i > 0:
                        left_start = i + 1
                        break
                right_end = len(expr)
                for i in range(mul_idx + 1, len(expr)):
                    if expr[i] in '+-*/':
                        right_end = i
                        break
                
                left_part = expr[left_start:mul_idx].strip()
                right_part = expr[mul_idx+1:right_end].strip()
                
                left_val = float(left_part) if left_part else 1.0
                right_val = float(right_part) if right_part else 1.0
                result = left_val * right_val
                
                expr = expr[:left_start] + str(result) + expr[right_end:]
            else:
                # Handle division
                left_start = 0
                for i in range(div_idx - 1, -1, -1):
                    if expr[i] in '+-' and i > 0:
                        left_start = i + 1
                        break
                right_end = len(expr)
                for i in range(div_idx + 1, len(expr)):
                    if expr[i] in '+-*/':
                        right_end = i
                        break
                
                left_part = expr[left_start:div_idx].strip()
                right_part = expr[div_idx+1:right_end].strip()
                
                left_val = float(left_part) if left_part else 1.0
                right_val = float(right_part) if right_part else 1.0
                if right_val == 0:
                    return None
                result = left_val / right_val
                
                expr = expr[:left_start] + str(result) + expr[right_end:]
        
        # Now handle addition and subtraction
        if '+' in expr or (expr.count('-') > 0 and not expr.startswith('-')):
            parts = []
            current = ''
            sign = 1
            for char in expr:
                if char == '+':
                    if current:
                        parts.append(sign * float(current.strip()))
                    current = ''
                    sign = 1
                elif char == '-' and current:
                    if current:
                        parts.append(sign * float(current.strip()))
                    current = ''
                    sign = -1
                else:
                    current += char
            if current:
                parts.append(sign * float(current.strip()))
            return sum(parts)
        else:
            return float(expr)
    except Exception as e:
        return None

def parse_value(value_str, constants=None):
    """
    Enhanced value parser that handles expressions and units with proper normalization.
    
    Parameters:
    -----------
    value_str : str
        String containing value to parse (e.g., "SiTrackerBarrel_inner_rc - 6.2*mm")
    constants : dict, optional
        Dictionary of constants
        
    Returns:
    --------
    float or None
    """
    if value_str is None:
        return None
        
    if isinstance(value_str, (int, float)):
        return float(value_str)
    
    # Unit conversions (all to mm for length, rad for angles)
    unit_conversions = {
        'mm': 1.0,
        'cm': 10.0,
        'm': 1000.0,
        'rad': 1.0,
        'deg': np.pi/180.0,
        'mrad': 0.001
    }
    
    try:
        # Check if this is a direct constant reference
        if constants and value_str in constants:
            value = constants[value_str]
            if isinstance(value, (int, float)):
                return float(value)
            # If constant itself is an expression, parse it recursively
            return parse_value(str(value), constants)
            
        expr = str(value_str)
        
        # First, substitute constants - need to do this before unit normalization
        # to handle cases where constants contain units
        if constants:
            # Sort by length (longest first) to avoid partial replacements
            const_names = sorted(constants.keys(), key=len, reverse=True)
            for const_name in const_names:
                const_value = constants[const_name]
                # Convert constant value to string for substitution
                if isinstance(const_value, (int, float)):
                    const_str = str(const_value)
                else:
                    const_str = str(const_value)
                # Only replace if it's a complete word/match
                # Replace whole word matches of constant names
                pattern = r'\b' + re.escape(const_name) + r'\b'
                expr = re.sub(pattern, const_str, expr)
        
        # Handle parentheses before unit normalization (e.g., "(787.105+1.75)*mm")
        # First evaluate inner expressions if they're wrapped in parentheses followed by unit
        paren_unit_pattern = r'\(([^)]+)\)\s*\*\s*(' + '|'.join(re.escape(u) for u in unit_conversions.keys()) + r')'
        def eval_paren_unit(match):
            inner_expr = match.group(1)
            unit = match.group(2)
            # Evaluate inner expression
            inner_result = evaluate_math_expression(inner_expr)
            if inner_result is not None:
                # Convert to base unit
                factor = unit_conversions[unit]
                return str(inner_result * factor)
            return match.group(0)  # Return original if evaluation fails
        
        # Replace parenthesized expressions with units
        expr = re.sub(paren_unit_pattern, eval_paren_unit, expr)
        
        # Normalize all remaining units to a base unit (mm for length, rad for angles)
        # Process each unit type, converting to base unit
        for unit, factor in unit_conversions.items():
            # Find all occurrences of number*unit pattern (but not already parenthesized ones)
            pattern = r'([-+]?\d*\.?\d+)\s*\*\s*' + re.escape(unit)
            def replace_unit(match):
                value = float(match.group(1))
                # Convert to base unit
                converted = value * factor
                return str(converted)
            expr = re.sub(pattern, replace_unit, expr)
        
        # Evaluate the mathematical expression (now all units are normalized)
        result = evaluate_math_expression(expr)
        if result is not None:
            return result
            
        return None
    except Exception as e:
        # Don't print warning for every failed parse - many are expected
        return None




def parse_tracker_module(module, config, constants):
    """Parse a tracker module definition"""
    trd = module.find('trd')
    if trd is not None:
        x1 = parse_value(trd.get('x1'), constants)
        x2 = parse_value(trd.get('x2'), constants)
        z = parse_value(trd.get('z'), constants)
        
        if any(v is None for v in [x1, x2, z]):
            return None
            
        # Calculate cells based on average width
        avg_width = (x1 + x2) / 2
        cell_size = config.get_cell_size()
        
        return {
            'x1': x1,
            'x2': x2,
            'z': z,
            'pixels_x': int(avg_width / cell_size['x']),
            'pixels_y': int(z / cell_size['y'])
        }
    
    # Try module_envelope for barrel-style modules
    envelope = module.find('module_envelope')
    if envelope is not None:
        width = parse_value(envelope.get('width'), constants)
        length = parse_value(envelope.get('length'), constants)
        
        if any(v is None for v in [width, length]):
            return None
            
        cell_size = config.get_cell_size()
        return {
            'width': width,
            'length': length,
            'pixels_x': int(width / cell_size['x']),
            'pixels_y': int(length / cell_size['y'])
        }
        
    return None

def parse_endcap_tracker_ring(ring, config, constants):
    """Parse a tracker endcap ring"""
    r_str = ring.get('r')
    zstart_str = ring.get('zstart')
    nmodules = int(ring.get('nmodules'))
    
    r = parse_value(r_str, constants)
    z = parse_value(zstart_str, constants)
    
    if any(v is None for v in [r, z]):
        return None
        
    return {
        'r': r,
        'z': z,
        'nmodules': nmodules
    }


def derive_tracker_envelope_bounds(layer_info, constants, config):
    """
    Derive envelope bounds for tracker layers when direct parsing fails.
    
    Parameters:
    -----------
    layer_info : dict
        Layer information dictionary (may already have some fields)
    constants : dict
        Constants dictionary
    config : DetectorConfig
        Detector configuration
        
    Returns:
    --------
    dict with 'inner_r', 'outer_r', 'z_length' if derivable, otherwise empty dict
    """
    derived = {}
    
    # Try to derive radial bounds from rc and module dimensions
    rc = layer_info.get('rc')
    width = layer_info.get('width')
    
    if rc is not None:
        # Try to get radial extent from constants or module width
        if width is not None:
            # Estimate radial extent from module width
            # For barrel trackers, modules are arranged around cylinder
            # Use half module width plus some tolerance
            half_width = width / 2.0
            # Common offsets seen in tracker XMLs: ~6.2mm inner, ~45mm outer
            inner_offset = 6.2  # Default fallback
            outer_offset = 45.0  # Default fallback
            
            # Try to get more precise offsets from constants
            detector_name = config.name if hasattr(config, 'name') else ''
            if detector_name:
                rc_dr = constants.get(f'{detector_name}_rc_dr')
                if rc_dr is not None:
                    inner_offset = rc_dr if isinstance(rc_dr, (int, float)) else parse_value(str(rc_dr), constants) or 6.2
                # Outer offset is typically larger
                outer_offset = max(inner_offset * 7, 45.0)  # Rough estimate
            
            derived['inner_r'] = rc - half_width - inner_offset
            derived['outer_r'] = rc + half_width + outer_offset
        else:
            # If no module width, try to estimate from constants
            detector_name = config.name if hasattr(config, 'name') else ''
            if detector_name and constants:
                # Try to get layer-specific constants
                rc_key = f'{detector_name}_inner_rc'
                if rc_key in constants:
                    base_rc = parse_value(str(constants[rc_key]), constants)
                    if base_rc:
                        inner_offset = 6.2
                        outer_offset = 45.0
                        derived['inner_r'] = base_rc - inner_offset
                        derived['outer_r'] = base_rc + outer_offset
    
    # Try to derive z_length from z0, nz, and module parameters
    z0 = layer_info.get('z0')
    nz = layer_info.get('nz')
    length = layer_info.get('length')
    
    if z0 is not None:
        if nz is not None and length is not None:
            # Estimate z_length from module layout
            # z_length = 2 * (z0 + module_length/2 + tolerance)
            module_z_spacing = constants.get('SiTracker_module_z_spacing')
            if module_z_spacing is not None:
                module_z_spacing = parse_value(str(module_z_spacing), constants) or 42.2
            else:
                module_z_spacing = 42.2  # Default from XML
            
            # Estimate total z extent
            z_extent = z0 + (nz - 1) * module_z_spacing / 2 if nz > 1 else z0
            tolerance = 60.0  # Common tolerance seen in tracker XMLs
            derived['z_length'] = 2 * (z_extent + length / 2 + tolerance)
        elif z0 is not None:
            # Fallback: estimate from z0 alone
            tolerance = 60.0
            estimated_length = length if length else 100.0  # Default module length
            derived['z_length'] = 2 * (z0 + estimated_length / 2 + tolerance)
    
    return derived


def parse_barrel_geometry(detector, config, geometry_info, constants):
    """Parse barrel-type detector geometry"""
    # First parse modules
    modules = {}
    for module in detector.findall(".//module"):
        name = module.get('name')
        if name is None:
            continue
            
        env = module.find('module_envelope')
        if env is not None:
            width = parse_value(env.get('width'), constants)
            length = parse_value(env.get('length'), constants)
            
            if width is not None and length is not None:
                cell_size = config.get_cell_size()
                pixels_x = int(width / cell_size['x'])
                pixels_y = int(length / cell_size['y'])
                
                modules[name] = {
                    'width': width,
                    'length': length,
                    'pixels_x': pixels_x,
                    'pixels_y': pixels_y,
                    'total_pixels': pixels_x * pixels_y
                }
    
    # Parse layers
    for layer in detector.findall('.//layer'):
        layer_id_str = layer.get('id')
        if layer_id_str is None:
            continue
            
        try:
            layer_id = int(layer_id_str)
        except ValueError:
            continue
            
        layer_info = {}
        
        # Get repeat count
        repeat = int(parse_value(layer.get('repeat', '1'), constants) or 1)
        
        # Process basic layer info
        module_name = layer.get('module')
        if module_name in modules:
            module = modules[module_name]
            layer_info.update(module)
        
        # Get barrel envelope
        barrel_env = layer.find('barrel_envelope')
        envelope_parsed = False
        if barrel_env is not None:
            inner_r = parse_value(barrel_env.get('inner_r'), constants)
            outer_r = parse_value(barrel_env.get('outer_r'), constants)
            z_length = parse_value(barrel_env.get('z_length'), constants)
            
            if all(v is not None for v in [inner_r, outer_r, z_length]):
                layer_info.update({
                    'inner_r': inner_r,
                    'outer_r': outer_r,
                    'z_length': z_length,
                    'rc': (inner_r + outer_r) / 2
                })
                envelope_parsed = True
        
        # Get rphi layout
        rphi = layer.find('rphi_layout')
        if rphi is not None:
            nphi_val = rphi.get('nphi')
            if nphi_val in constants:
                n_phi = int(float(constants[nphi_val]))
            else:
                n_phi = int(parse_value(nphi_val, constants) or 0)
                
            rc = parse_value(rphi.get('rc'), constants)
            if rc is not None:
                layer_info['rc'] = rc
                
            layer_info.update({
                'nphi': n_phi,
                'phi_tilt': parse_value(rphi.get('phi_tilt'), constants) or 0.0,
                'phi0': parse_value(rphi.get('phi0'), constants) or 0.0
            })
        
        # Get z layout
        z_layout = layer.find('z_layout')
        if z_layout is not None:
            nz_val = z_layout.get('nz')
            if nz_val in constants:
                n_z = int(float(constants[nz_val]))
            else:
                n_z = int(parse_value(nz_val, constants) or 1)
                
            layer_info.update({
                'nz': n_z,
                'z0': parse_value(z_layout.get('z0'), constants) or 0.0
            })
        
        # Calculate total cells
        if 'nphi' in layer_info and 'total_pixels' in layer_info:
            n_phi = layer_info['nphi']
            n_z = layer_info.get('nz', 1)
            pixels_per_module = layer_info['total_pixels']
            
            layer_info.update({
                'total_modules': n_phi * n_z,
                'total_cells': n_phi * n_z * pixels_per_module * repeat
            })
            
            # Add to overall total
            if 'total_cells' in layer_info:
                geometry_info['total_cells'] += layer_info['total_cells']
        
        # If envelope bounds are missing (especially for trackers), try to derive them
        if not envelope_parsed and config.detector_class.lower() in ['tracker', 'vertex']:
            derived_bounds = derive_tracker_envelope_bounds(layer_info, constants, config)
            # Only update if we successfully derived bounds and they're still missing
            for key in ['inner_r', 'outer_r', 'z_length']:
                if key not in layer_info and key in derived_bounds:
                    layer_info[key] = derived_bounds[key]
            # If we derived envelope bounds but rc wasn't set from envelope, calculate it
            if 'inner_r' in layer_info and 'outer_r' in layer_info:
                if 'rc' not in layer_info:
                    # Calculate rc from derived envelope bounds
                    layer_info['rc'] = (layer_info['inner_r'] + layer_info['outer_r']) / 2
                # If rc exists but was from rphi_layout, we keep it (it's more precise)
        
        # Store layer info
        geometry_info['layers'][layer_id] = layer_info


def parse_endcap_geometry(detector, config, geometry_info, constants):
    """
    Parse endcap-type detector geometry with robust error handling.
    
    Parameters:
    -----------
    detector : xml.etree.ElementTree.Element
        Detector XML element
    config : DetectorConfig
        Detector configuration
    geometry_info : dict
        Dictionary to store geometry information
    constants : dict
        Constants dictionary for value parsing
    """
    # First parse all module definitions
    modules = {}
    for module in detector.findall(".//module"):
        name = module.get('name')
        if name is None:
            continue
        
        # Try trd (trapezoid) definition first
        trd = module.find('trd')
        if trd is not None:
            # Parse trapezoid dimensions with error handling
            x1 = parse_value(trd.get('x1'), constants)
            x2 = parse_value(trd.get('x2'), constants)
            z = parse_value(trd.get('z'), constants)
            
            if all(v is not None for v in [x1, x2, z]):
                # Get cell size from config
                cell_size = config.get_cell_size()
                
                # Calculate average width for pixel counting
                avg_width = (x1 + x2) / 2
                pixels_x = int(avg_width / cell_size['x'])
                pixels_y = int(z / cell_size['y'])
                total_pixels = pixels_x * pixels_y
                
                modules[name] = {
                    'type': 'trd',
                    'x1': x1,
                    'x2': x2,
                    'z': z,
                    'width': avg_width,  # Use average width for calculations
                    'length': z,
                    'pixels_x': pixels_x,
                    'pixels_y': pixels_y,
                    'total_pixels': total_pixels
                }
                continue
        
        # Try module_envelope if no trd
        env = module.find('module_envelope')
        if env is not None:
            width = parse_value(env.get('width'), constants)
            length = parse_value(env.get('length'), constants)
            
            if width is not None and length is not None:
                cell_size = config.get_cell_size()
                pixels_x = int(width / cell_size['x'])
                pixels_y = int(length / cell_size['y'])
                total_pixels = pixels_x * pixels_y
                
                modules[name] = {
                    'type': 'envelope',
                    'width': width,
                    'length': length,
                    'pixels_x': pixels_x,
                    'pixels_y': pixels_y,
                    'total_pixels': total_pixels
                }
    
    # Parse layers
    for layer in detector.findall('.//layer'):
        # Get layer ID with error handling
        layer_id_str = layer.get('id')
        if layer_id_str is None:
            print(f"Warning: Layer found without id attribute")
            continue
            
        try:
            layer_id = int(layer_id_str)
        except ValueError:
            print(f"Warning: Invalid layer id: {layer_id_str}")
            continue
        
        # Get repeat count
        repeat = int(parse_value(layer.get('repeat', '1'), constants) or 1)
        
        layer_info = {
            'rings': [],  # Store ring information
            'total_cells': 0  # Initialize cell count for this layer
        }
        
        # Process all rings in this layer
        for ring in layer.findall('ring'):
            ring_info = {}
            
            # Get module type for this ring
            module_name = ring.get('module')
            if module_name not in modules:
                print(f"Warning: Module {module_name} not found for ring in layer {layer_id}")
                continue
            
            # Get module info
            module = modules[module_name]
            ring_info.update(module)  # Copy module properties
            
            # Parse ring-specific parameters with error handling
            r_val = ring.get('r')
            r = parse_value(r_val, constants)
            
            zstart_val = ring.get('zstart')
            zstart = parse_value(zstart_val, constants)
            
            # Handle nmodules which might be a constant
            nmodules_val = ring.get('nmodules')
            if nmodules_val in constants:
                nmodules = int(float(constants[nmodules_val]))
            else:
                try:
                    nmodules = int(parse_value(nmodules_val, constants) or 0)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid nmodules value: {nmodules_val}")
                    nmodules = 0
            
            # Get optional parameters
            phi0 = parse_value(ring.get('phi0'), constants) or 0.0
            dz = parse_value(ring.get('dz'), constants) or 0.0
            
            # Only add ring if we have valid basic parameters
            if all(v is not None for v in [r, zstart]) and nmodules > 0:
                ring_info.update({
                    'r': r,
                    'zstart': zstart,
                    'nmodules': nmodules,
                    'phi0': phi0,
                    'dz': dz,
                    'total_cells': nmodules * module['total_pixels'] * repeat
                })
                
                # Calculate ring bounds using correct orientation
                if ring_info['type'] == 'trd':
                    # Get TRD orientation for this detector
                    orientation = _get_trd_orientation(config)
                    
                    if orientation == 'radial_z':
                        # For SiD-style disk detectors: z is radial, x1/x2 are azimuthal
                        radial_half_thickness = ring_info['z']
                        ring_info.update({
                            'r_inner': r - radial_half_thickness,
                            'r_outer': r + radial_half_thickness,
                            'z_min': zstart,
                            'z_max': zstart + radial_half_thickness,  # z extent in beam direction
                            'azimuthal_width_inner': 2 * ring_info['x1'],  # store for area calculations
                            'azimuthal_width_outer': 2 * ring_info['x2'],
                            'radial_thickness': 2 * radial_half_thickness
                        })
                    else:
                        # Historical behavior: x1/x2 are radial dimensions
                        ring_info.update({
                            'r_inner': r - ring_info['x1']/2,
                            'r_outer': r + ring_info['x2']/2,
                            'z_min': zstart,
                            'z_max': zstart + ring_info['z'],
                            'radial_thickness': ring_info['z']
                        })
                else:
                    half_width = ring_info['width'] / 2
                    ring_info.update({
                        'r_inner': r - half_width,
                        'r_outer': r + half_width,
                        'z_min': zstart,
                        'z_max': zstart + ring_info['length']
                    })
                
                layer_info['rings'].append(ring_info)
                layer_info['total_cells'] += ring_info['total_cells']
        
        # Only add layer if it has valid rings
        if layer_info['rings']:
            geometry_info['layers'][layer_id] = layer_info
            geometry_info['total_cells'] += layer_info['total_cells']
            
            # Store some layer-level summary info
            n_rings = len(layer_info['rings'])
            total_modules = sum(ring['nmodules'] for ring in layer_info['rings'])
            r_min = min(ring['r_inner'] for ring in layer_info['rings'])
            r_max = max(ring['r_outer'] for ring in layer_info['rings'])
            z_min = min(ring['z_min'] for ring in layer_info['rings'])
            z_max = max(ring['z_max'] for ring in layer_info['rings'])
            
            layer_info.update({
                'n_rings': n_rings,
                'total_modules': total_modules,
                'r_min': r_min,
                'r_max': r_max,
                'z_min': z_min,
                'z_max': z_max,
                # Map to standard envelope field names for consistency
                'inner_r': r_min,
                'outer_r': r_max
            })
            
            # For endcap detectors, z_min/z_max are more appropriate than z_length
            # but some code may expect z_length, so calculate it for compatibility
            if z_min is not None and z_max is not None:
                layer_info['z_length'] = z_max - z_min
                # Store z center position as well
                layer_info['z_center'] = (z_min + z_max) / 2
        else:
            print(f"Warning: No valid rings found for layer {layer_id}")

def find_matching_ring(hit_pos, layer_info):
    """
    Find which ring a hit belongs to, with proper bounds checking.
    
    Parameters:
    -----------
    hit_pos : tuple
        (x, y, z) position of hit
    layer_info : dict
        Layer geometry information containing rings
        
    Returns:
    --------
    dict : ring information or None if no match
    """
    x, y, z = hit_pos
    r = np.sqrt(x*x + y*y)
    
    # Quick check using layer bounds
    if not (layer_info['r_min'] <= r <= layer_info['r_max'] and
            layer_info['z_min'] <= abs(z) <= layer_info['z_max']):
        return None
    
    # Check each ring's bounds
    for ring in layer_info['rings']:
        if (ring['r_inner'] <= r <= ring['r_outer'] and 
            ring['z_min'] <= abs(z) <= ring['z_max']):
            return ring
            
    return None




def calculate_endcap_ring_bounds(ring_info, config=None):
    """
    Calculate the geometric bounds of a ring for hit assignment.
    
    Parameters:
    -----------
    ring_info : dict
        Ring geometry information
    config : DetectorConfig, optional
        Detector configuration for orientation detection
        
    Returns:
    --------
    dict with ring bounds (r_inner, r_outer, z_min, z_max)
    """
    r = ring_info['r']
    z = ring_info['zstart']
    
    # If we have a trapezoid module, use correct orientation
    if ring_info['type'] == 'trd':
        if config:
            orientation = _get_trd_orientation(config)
        else:
            orientation = ring_info.get('orientation', 'radial_x')
            
        if orientation == 'radial_z':
            # For SiD-style: z is radial dimension
            radial_half_thickness = ring_info['z']
            r_inner = r - radial_half_thickness
            r_outer = r + radial_half_thickness
            z_extent = radial_half_thickness  # minimal z extent
        else:
            # Historical: x1/x2 are radial
            r_inner = r - ring_info['x1']/2
            r_outer = r + ring_info['x2']/2
            z_extent = ring_info['z']
    else:
        # For rectangular modules, use width/2 in both directions
        half_width = ring_info['width'] / 2
        r_inner = r - half_width
        r_outer = r + half_width
        z_extent = ring_info['length']
    
    return {
        'r_inner': r_inner,
        'r_outer': r_outer,
        'z_min': z,
        'z_max': z + z_extent
    }



def find_readout_info(root, detector_name):
    """Find readout information for a detector in the XML tree"""
    readout_name = f"{detector_name}Hits"
    readout = root.find(f".//readout[@name='{readout_name}']")
    
    if readout is not None:
        segmentation = readout.find(".//segmentation[@type='CartesianGridXY']")
        if segmentation is not None:
            return {
                'grid_size_x': segmentation.get('grid_size_x'),
                'grid_size_y': segmentation.get('grid_size_y')
            }
    return None


def process_calo_hit(hit_data, layer_info, is_barrel=True):
    """
    Process a calorimeter hit to determine its cell location.
    
    Parameters:
    -----------
    hit_data : dict
        Hit information including position and decoded cellID
    layer_info : dict
        Layer geometry information
    is_barrel : bool
        Whether this is a barrel calorimeter
        
    Returns:
    --------
    tuple : hit key for occupancy counting
    """
    layer = hit_data['cellid_decoded']['layer']
    hit_pos = hit_data['pos']
    
    try:
        # Get cell indices
        cell_indices = get_calo_cell_coordinates(hit_pos, layer_info, is_barrel)
        return (layer,) + cell_indices
    except Exception as e:
        print(f"Warning: Error calculating cell indices: {str(e)}")
        return (layer, 0, 0)  # Fallback
    

def parse_forward_geometry(detector, config, geometry_info, constants):
    """Parse forward calorimeter style detectors (BeamCal/LumiCal)."""
    dims = detector.find("dimensions")
    if dims is None:
        raise ValueError("No <dimensions> element found for forward detector")

    inner_r = parse_value(dims.get("inner_r"), constants)
    outer_r = parse_value(dims.get("outer_r"), constants)
    inner_z = parse_value(dims.get("inner_z"), constants)
    outer_z = parse_value(dims.get("outer_z"), constants)
    
    # If outer_z is not specified, try to get from constants or calculate from layers
    if outer_z is None:
        # Try to get from constants (e.g., LumiCal_zmax)
        detector_name = config.name if hasattr(config, 'name') else ''
        if detector_name:
            zmax_constant = constants.get(f'{detector_name}_zmax')
            if zmax_constant is not None:
                outer_z = parse_value(str(zmax_constant), constants)
        
        # If still None, calculate from layer thicknesses
        if outer_z is None and inner_z is not None:
            total_thickness = 0
            for layer in detector.findall('.//layer'):
                layer_info = parse_forward_calo_layer(layer, constants)
                if layer_info:
                    repeat = layer_info.get('repeat', 1)
                    total_thickness += layer_info.get('total_thickness', 0) * repeat
            if total_thickness > 0:
                outer_z = inner_z + total_thickness

    geometry_info.update({
        'inner_r': inner_r,
        'outer_r': outer_r,
        'inner_z': inner_z,
        'outer_z': outer_z
    })

    cell_size = config.get_cell_size()
    cell_area = cell_size['x'] * cell_size['y'] if cell_size else 1.0
    detector_area = math.pi * (outer_r ** 2 - inner_r ** 2) if inner_r is not None and outer_r is not None else None

    geometry_info['layers'] = {}
    total_cells = 0
    layer_index = 1

    for layer in detector.findall('.//layer'):
        repeat = parse_repeat_value(layer, constants)
        sensitive_slices = parse_layer_sensitive_slices(layer)

        cells_per_layer = 0
        if detector_area and cell_area > 0:
            cells_per_layer = int(detector_area / cell_area)
        cells_per_layer = max(cells_per_layer, 1)

        for r in range(repeat):
            layer_data = {
                'cells_per_layer': cells_per_layer,
                'sensitive_slices': sensitive_slices,
                'total_cells': cells_per_layer * max(sensitive_slices, 1),
                'repeat_group': layer_index - 1,
                'repeat_number': r,
                # Add envelope bounds from detector-level
                'inner_r': inner_r,
                'outer_r': outer_r
            }
            
            # Add z bounds - use outer_z if available, otherwise calculate from layer thickness
            if inner_z is not None:
                layer_data['z_min'] = inner_z
            if outer_z is not None:
                layer_data['z_max'] = outer_z
            elif inner_z is not None:
                # Calculate approximate z_max from inner_z and detector extent
                # This is a fallback - ideally outer_z should be in XML
                layer_data['z_max'] = inner_z + 1000.0  # Large fallback value
            
            geometry_info['layers'][layer_index] = layer_data
            total_cells += cells_per_layer * max(sensitive_slices, 1)
            layer_index += 1

    geometry_info['total_cells'] = total_cells


def parse_muon_geometry(detector, config, geometry_info, constants):
    """Parse muon barrel and endcap geometry using available constants."""
    detector_name = config.name
    cell_size = config.get_cell_size()
    area_per_cell = cell_size['x'] * cell_size['y'] if cell_size else 1.0

    geometry_info['layers'] = {}
    total_cells = 0
    layer_index = 1

    if detector_name == 'MuonBarrel':
        r_min = parse_value(constants.get('MuonBarrel_rmin'), constants)
        r_max = parse_value(constants.get('MuonBarrel_rmax'), constants)
        half_length = parse_value(constants.get('MuonBarrel_half_length'), constants)

        if None in (r_min, r_max, half_length):
            return

        circumference = 2 * math.pi * ((r_min + r_max) / 2)
        length = 2 * half_length

        cells_phi = max(int(circumference / cell_size['x']), 1)
        cells_z = max(int(length / cell_size['y']), 1)
        cells_per_layer = cells_phi * cells_z

        for layer in detector.findall('.//layer'):
            repeat = parse_repeat_value(layer, constants)
            sensitive_slices = parse_layer_sensitive_slices(layer)
            for r in range(repeat):
                geometry_info['layers'][layer_index] = {
                    'cells_per_layer': cells_per_layer,
                    'sensitive_slices': sensitive_slices,
                    'total_cells': cells_per_layer * max(sensitive_slices, 1),
                    'repeat_group': layer_index - 1,
                    'repeat_number': r,
                    # Add envelope bounds
                    'inner_r': r_min,
                    'outer_r': r_max,
                    'z_length': length
                }
                total_cells += cells_per_layer * max(sensitive_slices, 1)
                layer_index += 1

        geometry_info['total_cells'] = total_cells
        geometry_info['length'] = length
        geometry_info['inner_r'] = r_min
        geometry_info['outer_r'] = r_max

    elif detector_name == 'MuonEndcap':
        r_min = parse_value(constants.get('MuonEndcap_rmin'), constants)
        r_max = parse_value(constants.get('MuonEndcap_rmax'), constants)
        z_min = parse_value(constants.get('MuonEndcap_zmin'), constants)
        z_max = parse_value(constants.get('MuonEndcap_zmax'), constants)

        if None in (r_min, r_max, z_min, z_max):
            return

        detector_area = math.pi * (r_max ** 2 - r_min ** 2)
        cells_per_plate = max(int(detector_area / area_per_cell), 1)

        for layer in detector.findall('.//layer'):
            repeat = parse_repeat_value(layer, constants)
            sensitive_slices = parse_layer_sensitive_slices(layer)
            for r in range(repeat):
                geometry_info['layers'][layer_index] = {
                    'cells_per_layer': cells_per_plate,
                    'sensitive_slices': sensitive_slices,
                    'total_cells': cells_per_plate * max(sensitive_slices, 1),
                    'repeat_group': layer_index - 1,
                    'repeat_number': r,
                    # Add envelope bounds
                    'inner_r': r_min,
                    'outer_r': r_max,
                    'z_min': z_min,
                    'z_max': z_max
                }
                total_cells += cells_per_plate * max(sensitive_slices, 1)
                layer_index += 1

        geometry_info['total_cells'] = total_cells
        geometry_info['inner_r'] = r_min
        geometry_info['outer_r'] = r_max
        geometry_info['zmin'] = z_min
        geometry_info['zmax'] = z_max

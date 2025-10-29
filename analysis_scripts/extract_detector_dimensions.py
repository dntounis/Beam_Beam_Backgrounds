#!/usr/bin/env python3
"""
Extract actual detector dimensions using the corrected geometry parsing
and compare with reference table values.
"""

import sys
from pathlib import Path
import numpy as np

# Add the framework to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def extract_detector_dimensions():
    """Extract dimensions for all SiD detector systems."""
    
    try:
        from src.detector_config import get_detector_configs, get_xmls
        from src.geometry_parsing.geometry_info import get_geometry_info
        from src.geometry_parsing.k4geo_parsers import parse_detector_constants
        import xml.etree.ElementTree as ET
        
        # Get the paths to use the newer geometry
        detector_paths = {
            'main_xml': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/C3_bkg_studies/full_sim_studies_2024/dd4hep/lcgeo/SiD/compact/SiD_o2_v04/SiD_o2_v04.xml',
            'SiVertexBarrel': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/C3_bkg_studies/full_sim_studies_2024/dd4hep/lcgeo/SiD/compact/SiD_o2_v04/SiVertexBarrel_o2_v04.xml',
            'SiVertexEndcap': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/C3_bkg_studies/full_sim_studies_2024/dd4hep/lcgeo/SiD/compact/SiD_o2_v04/SiVertexEndcap_o2_v04.xml',
            'SiTrackerBarrel': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/C3_bkg_studies/full_sim_studies_2024/dd4hep/lcgeo/SiD/compact/SiD_o2_v04/SiTrackerBarrel_o2_v04.xml',
            'SiTrackerEndcap': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/C3_bkg_studies/full_sim_studies_2024/dd4hep/lcgeo/SiD/compact/SiD_o2_v04/SiTrackerEndcap_o2_v04.xml',
            'SiTrackerForward': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/C3_bkg_studies/full_sim_studies_2024/dd4hep/lcgeo/SiD/compact/SiD_o2_v04/SiTrackerForward_o2_v04.xml',
            'ECalBarrel': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/C3_bkg_studies/full_sim_studies_2024/dd4hep/lcgeo/SiD/compact/SiD_o2_v04/ECalBarrel_o2_v04.xml',
            'ECalEndcap': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/C3_bkg_studies/full_sim_studies_2024/dd4hep/lcgeo/SiD/compact/SiD_o2_v04/ECalEndcap_o2_v04.xml',
            'HCalBarrel': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/C3_bkg_studies/full_sim_studies_2024/dd4hep/lcgeo/SiD/compact/SiD_o2_v04/HCalBarrel_o2_v04.xml',
            'HCalEndcap': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/C3_bkg_studies/full_sim_studies_2024/dd4hep/lcgeo/SiD/compact/SiD_o2_v04/HCalEndcap_o2_v04.xml',
            'MuonBarrel': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/C3_bkg_studies/full_sim_studies_2024/dd4hep/lcgeo/SiD/compact/SiD_o2_v04/MuonBarrel_o2_v04.xml',
            'MuonEndcap': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/C3_bkg_studies/full_sim_studies_2024/dd4hep/lcgeo/SiD/compact/SiD_o2_v04/MuonEndcap_o2_v04.xml',
            'LumiCal': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/C3_bkg_studies/full_sim_studies_2024/dd4hep/lcgeo/SiD/compact/SiD_o2_v04/LumiCal_o2_v04.xml',
            'BeamCal': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/C3_bkg_studies/full_sim_studies_2024/dd4hep/lcgeo/SiD/compact/SiD_o2_v04/BeamCal_o2_v04.xml'
        }
        
        configs = get_detector_configs()
        main_xml = detector_paths['main_xml']
        
        results = {}
        
        print("Extracting detector dimensions using corrected geometry parsing...")
        print("=" * 80)
        
        for detector_name, xml_path in detector_paths.items():
            if detector_name == 'main_xml':
                continue
                
            if detector_name not in configs:
                print(f"Warning: No config found for {detector_name}")
                continue
                
            print(f"\nProcessing {detector_name}...")
            
            try:
                config = configs[detector_name]
                constants = parse_detector_constants(main_xml, detector_name)
                geometry_info = get_geometry_info(xml_path, config, constants=constants, main_xml=main_xml)
                
                # Extract dimensions based on detector type
                if config.detector_type == 'barrel':
                    dims = extract_barrel_dimensions(geometry_info, xml_path, constants)
                else:  # endcap
                    dims = extract_endcap_dimensions(geometry_info, xml_path, constants)
                
                results[detector_name] = {
                    'type': config.detector_type,
                    'class': config.detector_class,
                    'dimensions': dims
                }
                
                print(f"  Type: {config.detector_type}, Class: {config.detector_class}")
                if dims:
                    for key, value in dims.items():
                        if value is not None:
                            print(f"  {key}: {value:.1f} mm = {value/10:.1f} cm")
                        else:
                            print(f"  {key}: Not found")
                else:
                    print("  No dimensions extracted")
                    
            except Exception as e:
                print(f"  Error: {e}")
                results[detector_name] = {'error': str(e)}
        
        return results
        
    except Exception as e:
        print(f"Failed to extract dimensions: {e}")
        return {}

def extract_barrel_dimensions(geometry_info, xml_path, constants):
    """Extract r_inner, r_outer, z_range for barrel detectors."""
    
    dims = {
        'r_inner': None,
        'r_outer': None, 
        'z_half_length': None
    }
    
    # Try to get from constants first (more reliable)
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    detector_elem = tree.find('.//detector')
    detector_name = detector_elem.get('name', '')
    
    # Look for standard constants
    for prefix in [detector_name, detector_name.replace('Si', ''), detector_name.replace('Barrel', '')]:
        r_min_key = f'{prefix}_rmin'
        r_max_key = f'{prefix}_rmax' 
        half_length_key = f'{prefix}_half_length'
        z_length_key = f'{prefix}_z_length'
        
        if r_min_key in constants:
            dims['r_inner'] = constants[r_min_key]
        if r_max_key in constants:
            dims['r_outer'] = constants[r_max_key]
        if half_length_key in constants:
            dims['z_half_length'] = constants[half_length_key]
        elif z_length_key in constants:
            dims['z_half_length'] = constants[z_length_key] / 2
    
    # Try to extract from XML dimensions element
    dimensions = detector_elem.find('.//dimensions')
    if dimensions is not None:
        from src.geometry_parsing.k4geo_parsers import parse_value
        
        for attr in ['rmin', 'rmax', 'inner_r', 'outer_r']:
            value = parse_value(dimensions.get(attr), constants)
            if value is not None:
                if attr in ['rmin', 'inner_r'] and dims['r_inner'] is None:
                    dims['r_inner'] = value
                elif attr in ['rmax', 'outer_r'] and dims['r_outer'] is None:
                    dims['r_outer'] = value
        
        # Try z dimensions
        for attr in ['z', 'dz', 'half_length']:
            value = parse_value(dimensions.get(attr), constants)
            if value is not None and dims['z_half_length'] is None:
                dims['z_half_length'] = value
    
    # For tracker detectors, try to extract from layer information
    if geometry_info and 'layers' in geometry_info:
        r_values = []
        z_values = []
        
        for layer_info in geometry_info['layers'].values():
            # Get radial information
            if 'inner_r' in layer_info and layer_info['inner_r'] is not None:
                r_values.append(layer_info['inner_r'])
            if 'outer_r' in layer_info and layer_info['outer_r'] is not None:
                r_values.append(layer_info['outer_r'])
            if 'rc' in layer_info and layer_info['rc'] is not None:
                r_values.append(layer_info['rc'])
                
            # Get z information
            if 'z_length' in layer_info and layer_info['z_length'] is not None:
                z_values.append(layer_info['z_length'] / 2)
            if 'z0' in layer_info and layer_info['z0'] is not None:
                z_values.append(abs(layer_info['z0']))
        
        if r_values:
            if dims['r_inner'] is None:
                dims['r_inner'] = min(r_values)
            if dims['r_outer'] is None:
                dims['r_outer'] = max(r_values)
        
        if z_values and dims['z_half_length'] is None:
            dims['z_half_length'] = max(z_values)
    
    return dims

def extract_endcap_dimensions(geometry_info, xml_path, constants):
    """Extract z_inner, z_outer, r_outer for endcap detectors."""
    
    dims = {
        'z_inner': None,
        'z_outer': None,
        'r_outer': None,
        'r_inner': None  # Also track inner radius
    }
    
    # Try to get from constants first
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    detector_elem = tree.find('.//detector')
    detector_name = detector_elem.get('name', '')
    
    # Look for standard constants
    for prefix in [detector_name, detector_name.replace('Si', ''), detector_name.replace('Endcap', '').replace('Forward', '')]:
        z_min_key = f'{prefix}_zmin'
        z_max_key = f'{prefix}_zmax'
        r_min_key = f'{prefix}_rmin'
        r_max_key = f'{prefix}_rmax'
        inner_z_key = f'{prefix}_inner_z'
        outer_z_key = f'{prefix}_outer_z'
        inner_r_key = f'{prefix}_inner_r'
        outer_r_key = f'{prefix}_outer_r'
        
        if z_min_key in constants:
            dims['z_inner'] = constants[z_min_key]
        if z_max_key in constants:
            dims['z_outer'] = constants[z_max_key]
        if inner_z_key in constants:
            dims['z_inner'] = constants[inner_z_key]
        if outer_z_key in constants:
            dims['z_outer'] = constants[outer_z_key]
            
        if r_min_key in constants:
            dims['r_inner'] = constants[r_min_key]
        if r_max_key in constants:
            dims['r_outer'] = constants[r_max_key]
        if inner_r_key in constants:
            dims['r_inner'] = constants[inner_r_key]
        if outer_r_key in constants:
            dims['r_outer'] = constants[outer_r_key]
    
    # Try to extract from XML dimensions element
    dimensions = detector_elem.find('.//dimensions')
    if dimensions is not None:
        from src.geometry_parsing.k4geo_parsers import parse_value
        
        for attr, dim_key in [('zmin', 'z_inner'), ('zmax', 'z_outer'), 
                              ('inner_z', 'z_inner'), ('outer_z', 'z_outer'),
                              ('rmin', 'r_inner'), ('rmax', 'r_outer'),
                              ('inner_r', 'r_inner'), ('outer_r', 'r_outer')]:
            value = parse_value(dimensions.get(attr), constants)
            if value is not None and dims[dim_key] is None:
                dims[dim_key] = value
    
    # For tracker detectors, extract from layer/ring information
    if geometry_info and 'layers' in geometry_info:
        z_values = []
        r_values = []
        
        for layer_info in geometry_info['layers'].values():
            # For endcap trackers with rings
            if 'rings' in layer_info:
                for ring in layer_info['rings']:
                    if 'z_min' in ring and ring['z_min'] is not None:
                        z_values.append(ring['z_min'])
                    if 'z_max' in ring and ring['z_max'] is not None:
                        z_values.append(ring['z_max'])
                    if 'zstart' in ring and ring['zstart'] is not None:
                        z_values.append(ring['zstart'])
                        
                    if 'r_inner' in ring and ring['r_inner'] is not None:
                        r_values.append(ring['r_inner'])
                    if 'r_outer' in ring and ring['r_outer'] is not None:
                        r_values.append(ring['r_outer'])
                    if 'r' in ring and ring['r'] is not None:
                        r_values.append(ring['r'])
            
            # Also check layer-level bounds
            for key in ['z_min', 'z_max', 'zmin', 'zmax']:
                if key in layer_info and layer_info[key] is not None:
                    z_values.append(layer_info[key])
            
            for key in ['r_min', 'r_max', 'rmin', 'rmax']:
                if key in layer_info and layer_info[key] is not None:
                    r_values.append(layer_info[key])
        
        if z_values:
            if dims['z_inner'] is None:
                dims['z_inner'] = min(z_values)
            if dims['z_outer'] is None:
                dims['z_outer'] = max(z_values)
        
        if r_values:
            if dims['r_inner'] is None:
                dims['r_inner'] = min(r_values)
            if dims['r_outer'] is None:
                dims['r_outer'] = max(r_values)
    
    return dims

def compare_with_reference():
    """Compare extracted dimensions with reference table."""
    
    # Reference table values (in cm)
    reference_barrel = {
        'SiVertexBarrel': {'r_inner': 1.4, 'r_outer': 6.0, 'z_range': 6.25},
        'SiTrackerBarrel': {'r_inner': 21.7, 'r_outer': 122.1, 'z_range': 152.2},
        'ECalBarrel': {'r_inner': 126.5, 'r_outer': 140.9, 'z_range': 176.5},
        'HCalBarrel': {'r_inner': 141.7, 'r_outer': 249.3, 'z_range': 301.8},
        'MuonBarrel': {'r_inner': 340.2, 'r_outer': 604.2, 'z_range': 303.3}
    }
    
    reference_endcap = {
        'SiVertexEndcap': {'z_inner': 7.6, 'z_outer': 18.0, 'r_outer': 7.1},
        'SiTrackerForward': {'z_inner': 21.1, 'z_outer': 83.4, 'r_outer': 16.6},
        'SiTrackerEndcap': {'z_inner': 77.0, 'z_outer': 164.3, 'r_outer': 125.5},
        'ECalEndcap': {'z_inner': 165.7, 'z_outer': 180.0, 'r_outer': 125.0},
        'HCalEndcap': {'z_inner': 180.5, 'z_outer': 302.8, 'r_outer': 140.2},
        'MuonEndcap': {'z_inner': 303.3, 'z_outer': 567.3, 'r_outer': 604.2},
        'LumiCal': {'z_inner': 155.7, 'z_outer': 170.0, 'r_outer': 20.0},
        'BeamCal': {'z_inner': 277.5, 'z_outer': 300.7, 'r_outer': 13.5}
    }
    
    results = extract_detector_dimensions()
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH REFERENCE TABLE")
    print("=" * 80)
    
    print("\nBARREL DETECTORS:")
    print("-" * 80)
    print(f"{'Detector':<20} {'Parameter':<12} {'Extracted [cm]':<15} {'Reference [cm]':<15} {'Diff [%]':<10}")
    print("-" * 80)
    
    for detector, ref_data in reference_barrel.items():
        if detector in results and 'dimensions' in results[detector]:
            dims = results[detector]['dimensions']
            
            # Convert mm to cm
            r_inner_cm = dims['r_inner'] / 10 if dims['r_inner'] is not None else None
            r_outer_cm = dims['r_outer'] / 10 if dims['r_outer'] is not None else None  
            z_range_cm = dims['z_half_length'] / 10 if dims['z_half_length'] is not None else None
            
            # Compare each parameter
            comparisons = [
                ('r_inner', r_inner_cm, ref_data['r_inner']),
                ('r_outer', r_outer_cm, ref_data['r_outer']),
                ('z_range', z_range_cm, ref_data['z_range'])
            ]
            
            for param, extracted, reference in comparisons:
                if extracted is not None:
                    diff_pct = ((extracted - reference) / reference) * 100
                    print(f"{detector:<20} {param:<12} {extracted:<15.1f} {reference:<15.1f} {diff_pct:<+10.1f}")
                else:
                    print(f"{detector:<20} {param:<12} {'Not found':<15} {reference:<15.1f} {'N/A':<10}")
    
    print("\nENDCAP DETECTORS:")
    print("-" * 80)
    print(f"{'Detector':<20} {'Parameter':<12} {'Extracted [cm]':<15} {'Reference [cm]':<15} {'Diff [%]':<10}")
    print("-" * 80)
    
    for detector, ref_data in reference_endcap.items():
        if detector in results and 'dimensions' in results[detector]:
            dims = results[detector]['dimensions']
            
            # Convert mm to cm
            z_inner_cm = dims['z_inner'] / 10 if dims['z_inner'] is not None else None
            z_outer_cm = dims['z_outer'] / 10 if dims['z_outer'] is not None else None
            r_outer_cm = dims['r_outer'] / 10 if dims['r_outer'] is not None else None
            
            # Compare each parameter
            comparisons = [
                ('z_inner', z_inner_cm, ref_data['z_inner']),
                ('z_outer', z_outer_cm, ref_data['z_outer']),
                ('r_outer', r_outer_cm, ref_data['r_outer'])
            ]
            
            for param, extracted, reference in comparisons:
                if extracted is not None:
                    diff_pct = ((extracted - reference) / reference) * 100
                    print(f"{detector:<20} {param:<12} {extracted:<15.1f} {reference:<15.1f} {diff_pct:<+10.1f}")
                else:
                    print(f"{detector:<20} {param:<12} {'Not found':<15} {reference:<15.1f} {'N/A':<10}")

if __name__ == "__main__":
    compare_with_reference()
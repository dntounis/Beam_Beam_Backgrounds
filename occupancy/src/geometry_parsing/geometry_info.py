import xml.etree.ElementTree as ET

from src.geometry_parsing.k4geo_parsers import parse_barrel_geometry, parse_endcap_geometry, parse_forward_geometry, parse_muon_geometry, parse_calorimeter_geometry


def get_geometry_info(xml_file, config, constants=None, main_xml=None):
    """
    Parse geometry XML file for any detector type.
    
    Parameters:
    -----------
    xml_file : str
        Path to detector-specific XML file
    config : DetectorConfig
        Detector configuration
    constants : dict, optional
        Pre-parsed constants
    main_xml : str, optional
        Path to main XML file
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Initialize geometry info
        geometry_info = {
            'detector_name': config.name,
            'detector_type': config.detector_type,
            'layers': {},
            'total_cells': 0
        }
        
        # Find detector element
        detector = root.find(".//detector")
        if detector is None:
            raise ValueError(f"No detector element found in {xml_file}")
        
        # Get detector class and determine parsing approach
        detector_class = config.detector_class.lower()
        
        if detector_class in ['ecal', 'hcal']:
            # Use calorimeter-specific parsing
            parse_calorimeter_geometry(detector, config, geometry_info, constants)
        elif detector_class in ['vertex', 'tracker']:
            # Use existing tracker parsing based on detector type
            if config.detector_type == 'barrel':
                parse_barrel_geometry(detector, config, geometry_info, constants)
            else:
                parse_endcap_geometry(detector, config, geometry_info, constants)
        elif detector_class in ['beamcal', 'lumical']:
            # Forward calorimeters use simpler grid-based geometry
            parse_forward_geometry(detector, config, geometry_info, constants)
        elif detector_class == 'muon':
            # Muon system has its own geometry structure
            parse_muon_geometry(detector, config, geometry_info, constants)
        
        # Print geometry summary
        print(f"\nGeometry info for {config.name}:")
        print(f"Total cells: {geometry_info['total_cells']}")
        print("Layers:")
        for layer_id, layer_info in sorted(geometry_info['layers'].items()):
            print(f"  Layer {layer_id}:")
            for key, value in sorted(layer_info.items()):
                if isinstance(value, dict):
                    continue  # Skip nested dictionaries for clarity
                print(f"    {key}: {value}")
        
        return geometry_info
        
    except Exception as e:
        print(f"Error processing {config.name}: {str(e)}")
        raise

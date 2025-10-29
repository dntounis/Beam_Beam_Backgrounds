
class DetectorConfig:
    """Configuration class for different detector types"""
    
    # Default cell sizes in mm
    DEFAULT_CELL_SIZES = {
        'vertex': {'x': 0.01, 'y': 0.01},      # 10x10 micron
        'tracker': {'x': 0.02, 'y': 0.02},     # 20x20 micron
        'ecal': {'x': 3.0, 'y': 3.0},          # 3x3 mm
        'hcal': {'x': 30.0, 'y': 30.0},        # 3x3 cm
        'muon': {'x': 30.0, 'y': 30.0},        # 3x3 cm
        'beamcal': {'x': 3.5, 'y': 3.5},       # 3.5x3.5 mm
        'lumical': {'x': 3.5, 'y': 3.5}        # 3.5x3.5 mm
    }
    
    def __init__(self, name, detector_type, detector_class, cell_sizes=None):
        """
        Parameters:
        -----------
        name : str
            Detector name (e.g., 'SiVertexBarrel', 'ECalEndcap')
        detector_type : str
            Type of detector ('barrel' or 'endcap')
        detector_class : str
            Class of detector ('vertex', 'tracker', 'ecal', 'hcal')
        cell_sizes : dict, optional
            Override default cell sizes
        """
        self.name = name
        self.detector_type = detector_type.lower()
        self.detector_class = detector_class.lower()
        
        # Set default cell size based on detector class
        self.cell_sizes = {'default': self.DEFAULT_CELL_SIZES[self.detector_class]}
        if cell_sizes:
            self.cell_sizes.update(cell_sizes)
    
    def get_cell_size(self, layer=None, region=None):
        if layer is not None and layer in self.cell_sizes:
            return self.cell_sizes[layer]
        if region is not None and region in self.cell_sizes:
            return self.cell_sizes[region]
        return self.cell_sizes['default']


def get_detector_configs():
    """Returns a dictionary of detector configurations"""
    
    DETECTOR_CONFIGS = {
        # Existing vertex detector configs
        'SiVertexBarrel': DetectorConfig(
            name='SiVertexBarrel',
            detector_type='barrel',
            detector_class='vertex',
            cell_sizes={'default': {'x': 0.010, 'y': 0.010}}  # 3x3 micron
        ),
        'SiVertexEndcap': DetectorConfig(
            name='SiVertexEndcap',
            detector_type='endcap',
            detector_class='vertex',
            cell_sizes={'default': {'x': 0.010, 'y': 0.010}}
        ),
        
        # Add tracker detector configs
        'SiTrackerBarrel': DetectorConfig(
            name='SiTrackerBarrel',
            detector_type='barrel',
            detector_class='tracker',
            cell_sizes={'default': {'x': 0.02, 'y': 0.02}}  # 20x20 micron
        ),
        'SiTrackerEndcap': DetectorConfig(
            name='SiTrackerEndcap', 
            detector_type='endcap',
            detector_class='tracker',
            cell_sizes={'default': {'x': 0.02, 'y': 0.02}}
        ),
        'SiTrackerForward': DetectorConfig(
            name='SiTrackerForward',
            detector_type='forward', 
            detector_class='tracker',
            cell_sizes={'default': {'x': 0.02, 'y': 0.02}}
        ),
        
        # Configs for other detectors (calorimeters, etc)
        'ECalBarrel': DetectorConfig(
            name='ECalBarrel',
            detector_type='barrel',
            detector_class='ecal',
            cell_sizes={'default': {'x': 3.0, 'y': 3.0}}  # 3x3 mm
        ),
        'ECalEndcap': DetectorConfig(
            name='ECalEndcap',
            detector_type='endcap',
            detector_class='ecal',
            cell_sizes={'default': {'x': 3.0, 'y': 3.0}}
        ),
        'HCalBarrel': DetectorConfig(
            name='HCalBarrel',
            detector_type='barrel',
            detector_class='hcal',
            cell_sizes={'default': {'x': 30.0, 'y': 30.0}}  # 30x30 mm
        ),
        'HCalEndcap': DetectorConfig(
            name='HCalEndcap',
            detector_type='endcap',
            detector_class='hcal',
            cell_sizes={'default': {'x': 30.0, 'y': 30.0}}
        ),
        'BeamCal': DetectorConfig(
            name='BeamCal',
            detector_type='endcap',
            detector_class='beamcal',
            cell_sizes={'default': {'x': 3.5, 'y': 3.5}}  # 3.5x3.5 mm
        ),
        'LumiCal': DetectorConfig(
            name='LumiCal',
            detector_type='endcap',
            detector_class='lumical',
            cell_sizes={'default': {'x': 3.5, 'y': 3.5}}
        ),
        'MuonBarrel': DetectorConfig(
            name='MuonBarrel',
            detector_type='barrel',
            detector_class='muon',
            cell_sizes={'default': {'x': 30.0, 'y': 30.0}}  # 30x30 mm
        ),
        'MuonEndcap': DetectorConfig(
            name='MuonEndcap',
            detector_type='endcap',
            detector_class='muon',
            cell_sizes={'default': {'x': 30.0, 'y': 30.0}}
        )
    }

    return DETECTOR_CONFIGS


def get_xmls():
    xmls = {
        'main_xml': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/k4geo/SiD/compact/SiD_o2_v04/SiD_o2_v04.xml',
        'vertex_barrel_xml': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/k4geo/SiD/compact/SiD_o2_v04/SiVertexBarrel_o2_v04.xml',
        'vertex_endcap_xml': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/k4geo/SiD/compact/SiD_o2_v04/SiVertexEndcap_o2_v04.xml',
        'tracker_barrel_xml': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/k4geo/SiD/compact/SiD_o2_v04/SiTrackerBarrel_o2_v04.xml',
        'tracker_endcap_xml': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/k4geo/SiD/compact/SiD_o2_v04/SiTrackerEndcap_o2_v04.xml',
        'tracker_forward_xml': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/k4geo/SiD/compact/SiD_o2_v04/SiTrackerForward_o2_v04.xml',
        'ecal_barrel_xml': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/k4geo/SiD/compact/SiD_o2_v04/ECalBarrel_o2_v04.xml',
        'ecal_endcap_xml': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/k4geo/SiD/compact/SiD_o2_v04/ECalEndcap_o2_v04.xml',
        'hcal_barrel_xml': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/k4geo/SiD/compact/SiD_o2_v04/HCalBarrel_o2_v04.xml',
        'hcal_endcap_xml': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/k4geo/SiD/compact/SiD_o2_v04/HCalEndcap_o2_v04.xml',
        'beamcal_xml': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/k4geo/SiD/compact/SiD_o2_v04/BeamCal_o2_v04.xml',
        'lumical_xml': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/k4geo/SiD/compact/SiD_o2_v04/LumiCal_o2_v04.xml',
        'muon_barrel_xml': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/k4geo/SiD/compact/SiD_o2_v04/MuonBarrel_o2_v04.xml',
        'muon_endcap_xml': '/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/k4geo/SiD/compact/SiD_o2_v04/MuonEndcap_o2_v04.xml'
    }
    return xmls


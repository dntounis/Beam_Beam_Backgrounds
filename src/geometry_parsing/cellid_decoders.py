

class BitFieldElement:
    """Implementation of DD4hep's BitFieldElement for decoding cellID fields"""
    def __init__(self, name, offset, width):
        """
        Args:
            name: Field name
            offset: Bit offset
            width: Bit width (negative means signed)
        """
        self.name = name
        self.offset = offset
        self.width = abs(width)
        self.is_signed = width < 0
        
        # Create mask for this field
        self.mask = ((1 << self.width) - 1) << offset
        
        # Calculate min/max values for range checks
        if self.is_signed:
            self.min_val = (1 << (self.width - 1)) - (1 << self.width)
            self.max_val = (1 << (self.width - 1)) - 1
        else:
            self.min_val = 0
            self.max_val = (1 << self.width) - 1
            
    def value(self, cellid):
        """Extract this field's value from a cellID"""
        # Convert to integer if numpy type
        cellid = int(cellid)
        
        val = (cellid & self.mask) >> self.offset
        if self.is_signed and (val & (1 << (self.width - 1))):
            val -= (1 << self.width)
        return val
        
class BitFieldCoder:
    """Implementation of DD4hep's BitFieldCoder for cellID decoding"""
    def __init__(self, descriptor):
        """
        Args:
            descriptor: Field descriptor string e.g. "system:5,side:-2,layer:6,module:11,sensor:8"
        """
        self.fields = []
        self.field_map = {}
        
        # Parse descriptor string
        offset = 0
        for field_desc in descriptor.split(','):
            parts = field_desc.strip().split(':')
            
            if len(parts) == 2:
                # Just name:width
                name = parts[0]
                width = int(parts[1])
                this_offset = offset
                offset += abs(width)
            elif len(parts) == 3:
                # name:offset:width 
                name = parts[0]
                this_offset = int(parts[1])
                width = int(parts[2])
                offset = this_offset + abs(width)
            else:
                raise ValueError(f"Invalid field descriptor: {field_desc}")
                
            # Create and store field
            field = BitFieldElement(name, this_offset, width)
            self.fields.append(field)
            self.field_map[name] = len(self.fields) - 1
            
    def get_field(self, name):
        """Get a field by name"""
        if name not in self.field_map:
            raise KeyError(f"Unknown field: {name}")
        return self.fields[self.field_map[name]]
    
    def decode(self, cellid):
        """Decode all fields from a cellID"""
        # Convert to integer if numpy type
        cellid = int(cellid)
        return {field.name: field.value(cellid) for field in self.fields}
    
def create_decoder(detector_type):
    """Create appropriate decoder for given detector type"""
    if detector_type in ["SiVertexBarrel", "SiVertexEndcap", "TrackerBarrel", "TrackerEndcap"]:
        return BitFieldCoder("system:5,side:-2,layer:6,module:11,sensor:8")
    elif detector_type in ["ECalBarrel", "ECalEndcap", "HCalBarrel", "HCalEndcap", "MuonBarrel", "MuonEndcap"]:
        return BitFieldCoder("system:5,side:-2,module:8,stave:4,layer:9,submodule:4,x:32:-16,y:-16")
    elif detector_type in ["BeamCal", "LumiCal"]:
        return BitFieldCoder("system:8,barrel:3,layer:8,slice:8,x:32:-16,y:-16")
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def decode_dd4hep_cellid(cellID, detector):
    """
    Decode a DD4hep cellID according to the readout definitions used by SiD.
    
    Args:
        cellID: Integer cellID to decode
        detector: String detector name (e.g. "SiVertexBarrel", "ECalBarrel", etc.)
        
    Returns:
        Dictionary with decoded fields
    """
    # Create decoder based on detector type
    if detector in ["SiVertexBarrel", "SiVertexEndcap", "TrackerBarrel", "TrackerEndcap", 
                   "SiTrackerBarrel", "SiTrackerEndcap", "SiTrackerForward"]:
        decoder = BitFieldCoder("system:5,side:-2,layer:6,module:11,sensor:8")
        
    elif detector in ["ECalBarrel", "ECalEndcap", "HCalBarrel", "HCalEndcap", 
                     "MuonBarrel", "MuonEndcap"]:
        decoder = BitFieldCoder("system:5,side:-2,module:8,stave:4,layer:9,submodule:4,x:32:-16,y:-16")
        
    elif detector in ["BeamCal", "LumiCal"]:
        decoder = BitFieldCoder("system:8,barrel:3,layer:8,slice:8,x:32:-16,y:-16")
        
    else:
        raise ValueError(f"Unknown detector type: {detector}")
        
    try:
        # Convert to integer if numpy type
        cellID = int(cellID)
        return decoder.decode(cellID)
        
    except Exception as e:
        print(f"Error decoding cellID {cellID}: {e}")
        print(f"Binary: {format(int(cellID), '064b' if cellID > 0xFFFFFFFF else '032b')}")
        raise

class BitFieldElement:
    """Implementation of DD4hep's BitFieldElement for decoding cellID fields"""
    def __init__(self, name, offset, width):
        self.name = name
        self.offset = offset
        self.width = abs(width)
        self.is_signed = width < 0
        
        # Create mask for this field
        self.mask = ((1 << self.width) - 1) << offset
        
        # Calculate min/max values for range checks
        if self.is_signed:
            self.min_val = (1 << (self.width - 1)) - (1 << self.width)
            self.max_val = (1 << (self.width - 1)) - 1
        else:
            self.min_val = 0
            self.max_val = (1 << self.width) - 1
            
    def value(self, cellid):
        """Extract this field's value from a cellID"""
        # Convert to integer if numpy type
        cellid = int(cellid)
        
        val = (cellid & self.mask) >> self.offset
        if self.is_signed and (val & (1 << (self.width - 1))):
            val -= (1 << self.width)
        return val

class BitFieldCoder:
    """Implementation of DD4hep's BitFieldCoder for cellID decoding"""
    def __init__(self, descriptor):
        self.fields = []
        self.field_map = {}
        
        # Parse descriptor string
        offset = 0
        for field_desc in descriptor.split(','):
            parts = field_desc.strip().split(':')
            
            if len(parts) == 2:
                # Just name:width
                name = parts[0]
                width = int(parts[1])
                this_offset = offset
                offset += abs(width)
            elif len(parts) == 3:
                # name:offset:width 
                name = parts[0]
                this_offset = int(parts[1])
                width = int(parts[2])
                offset = this_offset + abs(width)
            else:
                raise ValueError(f"Invalid field descriptor: {field_desc}")
                
            # Create and store field
            field = BitFieldElement(name, this_offset, width)
            self.fields.append(field)
            self.field_map[name] = len(self.fields) - 1
    
    def decode(self, cellid):
        """Decode all fields from a cellID"""
        # Convert to integer if numpy type
        cellid = int(cellid)
        return {field.name: field.value(cellid) for field in self.fields}








def decode_cellid_tracker(cellid):
    """
    Decode tracker cellID based on the DD4hep layout:
    "system:5,side:-2,layer:6,module:11,sensor:8"
    
    Total bits = 32 (5+2+6+11+8)
    
    Args:
        cellid: Integer cellID value
        
    Returns:
        Dictionary with decoded fields
    """
    cellid = int(cellid)
    
    # Starting from least significant bits
    current_pos = 0
    
    # sensor: 8 bits
    sensor = cellid & 0xFF
    current_pos += 8
    
    # module: 11 bits
    module = (cellid >> current_pos) & 0x7FF
    current_pos += 11
    
    # layer: 6 bits
    layer = (cellid >> current_pos) & 0x3F
    current_pos += 6
    
    # side: 2 bits (signed)
    side_raw = (cellid >> current_pos) & 0x3
    # Convert to signed - if highest bit is set, it's negative
    side = side_raw - 4 if side_raw & 0x2 else side_raw
    current_pos += 2
    
    # system: 5 bits
    system = (cellid >> current_pos) & 0x1F
    
    return {
        'system': system,
        'side': side,
        'layer': layer,
        'module': module,
        'sensor': sensor
    }
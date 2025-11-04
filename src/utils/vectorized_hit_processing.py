"""
Vectorized hit processing utilities for DD4hep analysis.

Rewritten to align with DD4hep bitfield layouts and detector geometry so
that vectorized results match the traditional reference implementation.
"""

import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple

# Bitfield descriptors mirror the scalar decoder in
# src.geometry_parsing.cellid_decoders.decode_dd4hep_cellid
DESCRIPTOR_MAP: Dict[str, str] = {
    # Silicon trackers and vertex detectors
    'SiVertexBarrel': "system:5,side:-2,layer:6,module:11,sensor:8",
    'SiVertexEndcap': "system:5,side:-2,layer:6,module:11,sensor:8",
    'SiTrackerBarrel': "system:5,side:-2,layer:6,module:11,sensor:8",
    'SiTrackerEndcap': "system:5,side:-2,layer:6,module:11,sensor:8",
    'SiTrackerForward': "system:5,side:-2,layer:6,module:11,sensor:8",
    # Sampling calorimeters and muon system
    'ECalBarrel': "system:5,side:-2,module:8,stave:4,layer:9,submodule:4,x:32:-16,y:-16",
    'ECalEndcap': "system:5,side:-2,module:8,stave:4,layer:9,submodule:4,x:32:-16,y:-16",
    'HCalBarrel': "system:5,side:-2,module:8,stave:4,layer:9,submodule:4,x:32:-16,y:-16",
    'HCalEndcap': "system:5,side:-2,module:8,stave:4,layer:9,submodule:4,x:32:-16,y:-16",
    'MuonBarrel': "system:5,side:-2,module:8,stave:4,layer:9,submodule:4,x:32:-16,y:-16",
    'MuonEndcap': "system:5,side:-2,module:8,stave:4,layer:9,submodule:4,x:32:-16,y:-16",
    # Forward calorimetry
    'BeamCal': "system:8,barrel:3,layer:8,slice:8,x:32:-16,y:-16",
    'LumiCal': "system:8,barrel:3,layer:8,slice:8,x:32:-16,y:-16",
}


def _get_descriptor(detector_name: str) -> str:
    try:
        return DESCRIPTOR_MAP[detector_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported detector for vectorized decoding: {detector_name}") from exc


def _parse_descriptor(descriptor: str) -> List[Tuple[str, int, int]]:
    """Return list of (name, offset, width) from a DD4hep bitfield descriptor."""
    fields: List[Tuple[str, int, int]] = []
    offset = 0
    for token in descriptor.split(','):
        parts = token.strip().split(':')
        if len(parts) == 2:
            name, width_str = parts
            width = int(width_str)
            field_offset = offset
            offset += abs(width)
        elif len(parts) == 3:
            name, offset_str, width_str = parts
            field_offset = int(offset_str)
            width = int(width_str)
            offset = field_offset + abs(width)
        else:
            raise ValueError(f"Invalid bitfield token '{token}' in descriptor '{descriptor}'")
        fields.append((name, field_offset, width))
    return fields


def decode_cellids_vectorized(cellids: np.ndarray, detector_name: str) -> Dict[str, np.ndarray]:
    """Vectorized DD4hep cellID decoding."""
    descriptor = _get_descriptor(detector_name)
    fields = _parse_descriptor(descriptor)

    cellids = np.asarray(cellids, dtype=np.uint64)
    decoded: Dict[str, np.ndarray] = {}

    for name, offset, width in fields:
        nbits = abs(width)
        mask = (1 << nbits) - 1
        values = (cellids >> offset) & mask
        if width < 0 and nbits > 0:
            sign_bit = 1 << (nbits - 1)
            values = np.where(values & sign_bit, values - (1 << nbits), values)
        decoded[name] = values.astype(np.int32)

    return decoded


def _build_structured_array(components: Dict[str, np.ndarray]) -> np.ndarray:
    if not components:
        return np.empty(0, dtype=[('layer', np.int32)])
    length = len(next(iter(components.values())))
    dtype = [(name, np.int32) for name in components]
    structured = np.zeros(length, dtype=dtype)
    for name, values in components.items():
        structured[name] = np.asarray(values, dtype=np.int32)
    return structured


def _resolve_side_array(raw_side: Optional[np.ndarray], pos_z: np.ndarray) -> np.ndarray:
    """
    Map DD4hep side field values to 0 (+z) or 1 (-z).
    In real DD4hep files: side=1 for +z, side=-1 for -z.
    Falls back to the hit z-position when the side field is missing or ambiguous.
    """
    if raw_side is None:
        return (pos_z < 0).astype(np.int32)

    side_vals = np.asarray(raw_side, dtype=np.int32)
    
    # Initialize sides array based on z-position (fallback)
    sides = (pos_z < 0).astype(np.int32)
    
    # Handle negative side values: side=-1 or side=-2 means -z
    negative_mask = side_vals < 0
    sides[negative_mask] = 1
    
    # Handle positive side values: side=1 means +z, but verify with z-position
    positive_mask = side_vals > 0
    if np.any(positive_mask):
        # Positive side values typically map to +z, but verify consistency
        sides[positive_mask] = np.where(pos_z[positive_mask] >= 0, 0, 1).astype(np.int32)
    
    # Handle zero side values: rely on hit position
    zero_mask = side_vals == 0
    if np.any(zero_mask):
        sides[zero_mask] = (pos_z[zero_mask] < 0).astype(np.int32)
    
    return sides


def _clip_indices(values: np.ndarray, max_count: float) -> np.ndarray:
    max_count = max(1, int(round(max_count)))
    return np.clip(values, 0, max_count - 1)


def _fill_tracker_barrel(indices: np.ndarray, layer_info: Dict, cell_size: Dict,
                         modules: np.ndarray, pos_x: np.ndarray, pos_y: np.ndarray,
                         pos_z: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> None:
    if indices.size == 0:
        return
    nphi = layer_info.get('nphi')
    rc = layer_info.get('rc')
    if not nphi or rc is None:
        return
    phi0 = layer_info.get('phi0', 0.0)
    phi_tilt = layer_info.get('phi_tilt', 0.0)
    module_width = layer_info.get('width', layer_info.get('module_width'))
    module_length = layer_info.get('length', layer_info.get('module_length'))
    if module_width is None or module_length is None:
        return

    hits_x = pos_x[indices]
    hits_y = pos_y[indices]
    hits_z = pos_z[indices]
    module_vals = modules[indices]

    phi_hit = np.arctan2(hits_y, hits_x)
    module_phi = phi0 + (2.0 * np.pi * module_vals / nphi) + phi_tilt
    dphi = (phi_hit - module_phi + np.pi) % (2 * np.pi) - np.pi
    local_t = rc * dphi
    z0 = layer_info.get('z0', 0.0)
    local_z = hits_z - z0

    px = np.floor((local_t + module_width / 2.0) / cell_size['x']).astype(np.int32)
    py = np.floor((local_z + module_length / 2.0) / cell_size['y']).astype(np.int32)
    px = _clip_indices(px, module_width / cell_size['x'])
    py = _clip_indices(py, module_length / cell_size['y'])
    ix[indices] = px
    iy[indices] = py


def _fill_tracker_forward(indices: np.ndarray, layer_info: Dict, cell_size: Dict,
                          pos_x: np.ndarray, pos_y: np.ndarray,
                          ix: np.ndarray, iy: np.ndarray) -> None:
    if indices.size == 0:
        return
    module_width = layer_info.get('width', 100.0)
    module_length = layer_info.get('length', 100.0)

    hits_x = pos_x[indices]
    hits_y = pos_y[indices]
    px = np.floor((hits_x + module_width / 2.0) / cell_size['x']).astype(np.int32)
    py = np.floor((hits_y + module_length / 2.0) / cell_size['y']).astype(np.int32)
    px = _clip_indices(px, module_width / cell_size['x'])
    py = _clip_indices(py, module_length / cell_size['y'])
    ix[indices] = px
    iy[indices] = py


def _fill_tracker_endcap(indices: np.ndarray, layer_info: Dict, cell_size: Dict,
                         modules: np.ndarray, pos_x: np.ndarray, pos_y: np.ndarray,
                         pos_z: np.ndarray, ix: np.ndarray, iy: np.ndarray,
                         side_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    count = indices.size
    if count == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int32)

    rings = layer_info.get('rings', [])
    valid = np.zeros(count, dtype=bool)
    ring_ids = np.full(count, -1, dtype=np.int32)
    hits_x = pos_x[indices]
    hits_y = pos_y[indices]
    hits_z = pos_z[indices]
    module_vals = modules[indices]
    r_hit = np.sqrt(hits_x ** 2 + hits_y ** 2)

    for ring_idx, ring in enumerate(rings):
        r_inner = ring.get('r_inner', ring.get('r_min', 0.0))
        r_outer = ring.get('r_outer', ring.get('r_max', 0.0))
        z_min = ring.get('z_min', layer_info.get('z_min', 0.0))
        z_max = ring.get('z_max', layer_info.get('z_max', 0.0))
        
        # Check radial bounds first
        ring_mask = (r_hit >= r_inner) & (r_hit <= r_outer)
        if not np.any(ring_mask):
            continue
        
        # Check z bounds based on side: +z side (side=0) uses positive z, -z side (side=1) uses negative z
        side_mask = side_values[indices]
        plus_z_mask = (side_mask == 0) & ring_mask
        minus_z_mask = (side_mask == 1) & ring_mask
        
        # For +z side: check if hit_z is within ring z bounds
        plus_z_valid = np.zeros_like(ring_mask, dtype=bool)
        if np.any(plus_z_mask):
            plus_z_indices = np.where(plus_z_mask)[0]
            plus_z_z_values = hits_z[plus_z_indices]
            plus_z_valid[plus_z_indices] = (plus_z_z_values >= z_min) & (plus_z_z_values <= z_max)
        
        # For -z side: check if hit_z is within mirrored ring z bounds
        minus_z_valid = np.zeros_like(ring_mask, dtype=bool)
        if np.any(minus_z_mask):
            minus_z_indices = np.where(minus_z_mask)[0]
            minus_z_z_values = hits_z[minus_z_indices]
            minus_z_valid[minus_z_indices] = (minus_z_z_values >= -z_max) & (minus_z_z_values <= -z_min)
        
        # Combine valid masks
        ring_mask = plus_z_valid | minus_z_valid
        
        if not np.any(ring_mask):
            continue
        valid[ring_mask] = True
        ring_ids[ring_mask] = ring_idx

        nmodules = ring.get('nmodules', 0)
        if nmodules <= 0:
            continue
        phi0 = ring.get('phi0', 0.0)
        module_phi = phi0 + (2.0 * np.pi * (module_vals[ring_mask] % nmodules) / nmodules)
        ring_r = ring.get('r', (r_inner + r_outer) / 2.0)
        dx = hits_x[ring_mask] - ring_r * np.cos(module_phi)
        dy = hits_y[ring_mask] - ring_r * np.sin(module_phi)
        cos_phi = np.cos(-module_phi)
        sin_phi = np.sin(-module_phi)
        local_x = dx * cos_phi - dy * sin_phi
        local_y = dx * sin_phi + dy * cos_phi

        if ring.get('type') == 'trd':
            if 'azimuthal_width_inner' in ring and 'azimuthal_width_outer' in ring:
                module_width = 0.5 * (ring['azimuthal_width_inner'] + ring['azimuthal_width_outer'])
            else:
                module_width = ring.get('x1', 0.0) + ring.get('x2', 0.0)

            if 'radial_thickness' in ring:
                module_length = ring['radial_thickness']
            else:
                module_length = ring.get('z', 0.0)
        else:
            module_width = ring.get('width', layer_info.get('width'))
            module_length = ring.get('length', layer_info.get('length'))

        if module_width is None or module_length is None:
            continue
        if module_width <= 0:
            module_width = cell_size['x']
        if module_length <= 0:
            module_length = cell_size['y']

        px = np.floor((local_x + module_width / 2.0) / cell_size['x']).astype(np.int32)
        py = np.floor((local_y + module_length / 2.0) / cell_size['y']).astype(np.int32)
        px = _clip_indices(px, module_width / cell_size['x'])
        py = _clip_indices(py, module_length / cell_size['y'])
        ix[indices[ring_mask]] = px
        iy[indices[ring_mask]] = py

    return valid, ring_ids


def _compute_tracker_components(decoded: Dict[str, np.ndarray], positions: np.ndarray,
                                config, geometry_info: Optional[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    layers = decoded.get('layer')
    if layers is None:
        raise ValueError("Decoded cellIDs missing 'layer' field")
    layers = layers.astype(np.int32, copy=False)
    modules = decoded.get('module')
    if modules is None:
        modules = np.zeros_like(layers)
    modules = modules.astype(np.int32, copy=False)

    count = layers.size
    ix = np.zeros(count, dtype=np.int32)
    iy = np.zeros(count, dtype=np.int32)
    rings = np.zeros(count, dtype=np.int32)
    valid = np.ones(count, dtype=bool)

    pos_x = positions[:, 0]
    pos_y = positions[:, 1]
    pos_z = positions[:, 2]

    layer_map = geometry_info['layers'] if geometry_info and 'layers' in geometry_info else {}
    det_type = config.detector_type.lower()

    if det_type in ('endcap', 'forward'):
        raw_side = decoded.get('side')
        side_values = _resolve_side_array(raw_side, pos_z)
    else:
        side_values = np.zeros(count, dtype=np.int32)

    for layer_id in np.unique(layers):
        layer_indices = np.where(layers == layer_id)[0]
        if layer_indices.size == 0:
            continue
        layer_info = layer_map.get(int(layer_id)) if layer_map else None
        if layer_info is None:
            valid[layer_indices] = False
            continue
        cell_size = config.get_cell_size(layer=int(layer_id))
        if det_type == 'barrel':
            _fill_tracker_barrel(layer_indices, layer_info, cell_size, modules, pos_x, pos_y, pos_z, ix, iy)
        elif det_type == 'forward':
            _fill_tracker_forward(layer_indices, layer_info, cell_size, pos_x, pos_y, ix, iy)
        elif det_type == 'endcap':
            layer_valid, ring_ids = _fill_tracker_endcap(
                layer_indices, layer_info, cell_size, modules, pos_x, pos_y, pos_z, ix, iy, side_values
            )
            valid[layer_indices] = layer_valid
            rings[layer_indices] = ring_ids
        else:
            valid[layer_indices] = False

    rings = np.where(valid, np.where(rings >= 0, rings, 0), -1)
    components = _build_structured_array({
        'layer': layers,
        'ring': rings,
        'module': modules,
        'ix': ix,
        'iy': iy,
        'side': side_values,
    })
    return components, valid


def _compute_calo_muon_components(decoded: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    layers = decoded.get('layer')
    if layers is None:
        raise ValueError("Decoded cellIDs missing 'layer' field")
    layers = layers.astype(np.int32, copy=False)
    module = decoded.get('module', np.zeros_like(layers)).astype(np.int32, copy=False)
    stave = decoded.get('stave', np.zeros_like(layers)).astype(np.int32, copy=False)
    slice_ids = decoded.get('slice', np.zeros_like(layers)).astype(np.int32, copy=False)
    ix = decoded.get('x', np.zeros_like(layers)).astype(np.int32, copy=False)
    iy = decoded.get('y', np.zeros_like(layers)).astype(np.int32, copy=False)

    components = _build_structured_array({
        'layer': layers,
        'module': module,
        'stave': stave,
        'slice': slice_ids,
        'ix': ix,
        'iy': iy,
    })
    return components, np.ones(layers.size, dtype=bool)


def _compute_forward_calo_components(decoded: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    layers = decoded.get('layer')
    if layers is None:
        raise ValueError("Decoded cellIDs missing 'layer' field")
    layers = layers.astype(np.int32, copy=False)
    slice_ids = decoded.get('slice', np.zeros_like(layers)).astype(np.int32, copy=False)
    ix = decoded.get('x', np.zeros_like(layers)).astype(np.int32, copy=False)
    iy = decoded.get('y', np.zeros_like(layers)).astype(np.int32, copy=False)

    components = _build_structured_array({
        'layer': layers,
        'slice': slice_ids,
        'ix': ix,
        'iy': iy,
    })
    return components, np.ones(layers.size, dtype=bool)


def _fallback_components(decoded: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    layers = decoded.get('layer')
    if layers is None:
        raise ValueError("Decoded cellIDs missing 'layer' field")
    layers = layers.astype(np.int32, copy=False)
    module = decoded.get('module', np.zeros_like(layers)).astype(np.int32, copy=False)
    components = _build_structured_array({
        'layer': layers,
        'module': module,
    })
    return components, np.ones(layers.size, dtype=bool)


def _lookup_total_cells(geometry_info: Optional[Dict], layer: int) -> int:
    if not geometry_info or 'layers' not in geometry_info:
        return 0
    layer_dict = geometry_info['layers']
    if layer in layer_dict:
        return layer_dict[layer].get('total_cells', 0)
    if (layer + 1) in layer_dict:
        return layer_dict[layer + 1].get('total_cells', 0)
    if (layer - 1) in layer_dict:
        return layer_dict[layer - 1].get('total_cells', 0)
    return 0


def _empty_stats(buffer_depths: Sequence[int]) -> Dict[int, Dict[str, object]]:
    empty = {
        'per_layer': {},
        'overall_cells_hit': 0,
        'overall_cells_above_threshold': 0,
        'max_hits_per_cell': 0,
        'hit_distribution': {}
    }
    return {int(threshold): dict(empty) for threshold in buffer_depths}


def process_hits_vectorized(cellids: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                            detector_name: str, buffer_depths: Sequence[int],
                            geometry_info: Optional[Dict], config) -> Dict[int, Dict[str, object]]:
    """Compute per-threshold occupancy statistics using vectorized primitives."""
    cellids = np.asarray(cellids)
    if cellids.size == 0:
        return _empty_stats(buffer_depths)

    positions = np.column_stack((np.asarray(x, dtype=np.float64),
                                 np.asarray(y, dtype=np.float64),
                                 np.asarray(z, dtype=np.float64)))

    decoded = decode_cellids_vectorized(cellids, detector_name)
    det_class = config.detector_class.lower()
    det_type = config.detector_type.lower()

    if det_class in ('vertex', 'tracker'):
        components, valid_mask = _compute_tracker_components(decoded, positions, config, geometry_info)
    elif det_class in ('ecal', 'hcal', 'muon'):
        components, valid_mask = _compute_calo_muon_components(decoded)
    elif det_class in ('beamcal', 'lumical'):
        components, valid_mask = _compute_forward_calo_components(decoded)
    else:
        components, valid_mask = _fallback_components(decoded)

    if components.size == 0:
        return _empty_stats(buffer_depths)

    if valid_mask is not None:
        components = components[valid_mask]
    if components.size == 0:
        return _empty_stats(buffer_depths)

    unique_components, hit_counts = np.unique(components, return_counts=True)
    layers_for_keys = unique_components['layer'].astype(int)
    component_names = unique_components.dtype.names or ()
    side_arr = None
    if 'side' in component_names:
        side_arr = unique_components['side'].astype(int)
    is_side_sensitive = det_type in ('endcap', 'forward') and side_arr is not None
    stats: Dict[int, Dict[str, object]] = {}

    for threshold in buffer_depths:
        layer_stats: Dict[int, Dict[str, object]] = {}
        for layer in np.unique(layers_for_keys):
            layer_mask = layers_for_keys == layer
            counts = hit_counts[layer_mask]
            if counts.size == 0:
                continue
            total_hits = int(counts.sum())
            cells_hit = int(counts.size)
            cells_above = int(np.sum(counts >= threshold))
            max_hits = int(counts.max())
            mean_hits = float(counts.mean())
            total_cells = _lookup_total_cells(geometry_info, int(layer))

            layer_entry: Dict[str, object] = {
                'cells_hit': cells_hit,
                'total_hits': total_hits,
                'cells_above_threshold': cells_above,
                'max_hits': max_hits,
                'mean_hits': mean_hits,
            }

            if total_cells > 0:
                occupancy_total = cells_above / total_cells
            else:
                occupancy_total = 0.0

            if is_side_sensitive:
                layer_side = side_arr[layer_mask]
                plus_mask = layer_side == 0
                minus_mask = layer_side == 1

                counts_plus = counts[plus_mask]
                counts_minus = counts[minus_mask]

                cells_above_plus = int(np.sum(counts_plus >= threshold)) if counts_plus.size else 0
                cells_above_minus = int(np.sum(counts_minus >= threshold)) if counts_minus.size else 0
                total_hits_plus = int(counts_plus.sum()) if counts_plus.size else 0
                total_hits_minus = int(counts_minus.sum()) if counts_minus.size else 0

                occupancy_plus = (cells_above_plus / total_cells) if total_cells > 0 else 0.0
                occupancy_minus = (cells_above_minus / total_cells) if total_cells > 0 else 0.0

                if counts_plus.size or counts_minus.size:
                    occupancy_avg = (occupancy_plus + occupancy_minus) / 2.0
                else:
                    occupancy_avg = 0.0

                layer_entry.update({
                    'occupancy': occupancy_avg,
                    'occupancy_plus_z': occupancy_plus,
                    'occupancy_minus_z': occupancy_minus,
                    'cells_above_threshold_plus_z': cells_above_plus,
                    'cells_above_threshold_minus_z': cells_above_minus,
                    'total_hits_plus_z': total_hits_plus,
                    'total_hits_minus_z': total_hits_minus,
                })
            else:
                layer_entry['occupancy'] = occupancy_total

            layer_stats[int(layer)] = layer_entry

        overall_cells_hit = int(sum(v['cells_hit'] for v in layer_stats.values())) if layer_stats else 0
        overall_cells_above = int(sum(v['cells_above_threshold'] for v in layer_stats.values())) if layer_stats else 0
        max_hits_per_cell = int(max((v['max_hits'] for v in layer_stats.values()), default=0))
        if hit_counts.size:
            unique_counts, freq = np.unique(hit_counts, return_counts=True)
            hit_distribution = {int(k): int(v) for k, v in zip(unique_counts, freq)}
        else:
            hit_distribution = {}

        stats[int(threshold)] = {
            'per_layer': layer_stats,
            'overall_cells_hit': overall_cells_hit,
            'overall_cells_above_threshold': overall_cells_above,
            'max_hits_per_cell': max_hits_per_cell,
            'hit_distribution': hit_distribution,
        }

    return stats


def benchmark_hit_processing(cellids: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                             detector_name: str, hit_thresholds: Sequence[int],
                             geometry_info: Optional[Dict], config,
                             max_hits_to_test: int = 100000) -> Dict[str, float]:
    """Simple wall-clock benchmark comparing vectorized vs traditional pipeline."""
    import time  # Local import to avoid unnecessary overhead elsewhere

    size = min(len(cellids), max_hits_to_test)
    cellids = np.asarray(cellids[:size], dtype=np.uint64)
    x = np.asarray(x[:size], dtype=np.float64)
    y = np.asarray(y[:size], dtype=np.float64)
    z = np.asarray(z[:size], dtype=np.float64)

    print(f"Benchmarking hit processing with {size} hits...")
    print("=" * 60)

    start = time.time()
    vectorized_stats = process_hits_vectorized(cellids, x, y, z, detector_name, hit_thresholds, geometry_info, config)
    vec_time = time.time() - start

    print(f"Vectorized processing completed in {vec_time:.3f}s")
    summary = {
        'vectorized_time': vec_time,
        'vectorized_layers': len(vectorized_stats.get(hit_thresholds[0], {}).get('per_layer', {}))
    }

    return summary

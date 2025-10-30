"""Helpers for converting hit histograms into physical surface densities."""

from typing import Dict, List, Optional

import numpy as np


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_layer_geometry(geometry_info: Dict) -> List[Dict[str, float]]:
    """Extract per-layer geometric parameters in a uniform format.

    Returns a list of dictionaries. Each entry contains the keys required by
    the area helper functions and is tagged with ``type`` = ``'cylinder'`` or
    ``'disk'``.
    """

    layers: List[Dict[str, float]] = []

    for _, info in geometry_info.get('layers', {}).items():
        rings = info.get('rings') or []
        if rings:
            for ring in rings:
                r_inner = _safe_float(ring.get('r_inner', ring.get('r_min')))
                r_outer = _safe_float(ring.get('r_outer', ring.get('r_max')))
                if r_inner is None or r_outer is None:
                    continue

                z_min = _safe_float(ring.get('z_min', info.get('z_min')))
                z_max = _safe_float(ring.get('z_max', info.get('z_max')))
                thickness = _safe_float(ring.get('z'))
                if thickness is None and z_min is not None and z_max is not None:
                    thickness = abs(z_max - z_min)

                layers.append({
                    'type': 'disk',
                    'r_inner': r_inner,
                    'r_outer': r_outer,
                    'z_min': z_min,
                    'z_max': z_max,
                    'thickness': thickness,
                })
            continue

        radius = _safe_float(info.get('rc'))
        r_inner = _safe_float(info.get('inner_r'))
        r_outer = _safe_float(info.get('outer_r'))
        if radius is None:
            if r_inner is not None and r_outer is not None:
                radius = 0.5 * (r_inner + r_outer)

        length = _safe_float(info.get('z_length'))
        if length is None:
            length = _safe_float(info.get('length'))
        z_min = _safe_float(info.get('z_min'))
        z_max = _safe_float(info.get('z_max'))
        if length is None and z_min is not None and z_max is not None:
            length = abs(z_max - z_min)

        layers.append({
            'type': 'cylinder',
            'radius': radius,
            'r_inner': r_inner,
            'r_outer': r_outer,
            'length': length,
            'z_min': z_min,
            'z_max': z_max,
        })

    return layers


def filter_hits_to_geometry(
    r_vals: np.ndarray,
    z_vals: np.ndarray,
    layers: List[Dict[str, float]],
    tolerance: float = 1.0,
) -> np.ndarray:
    """Return a mask selecting hits that fall within known geometry envelopes.

    Parameters
    ----------
    r_vals : np.ndarray
        Radial positions (mm) of hits.
    z_vals : np.ndarray
        Longitudinal positions (mm) of hits.
    layers : list of dict
        Output from :func:`extract_layer_geometry` describing detector layers.
    tolerance : float, optional
        Symmetric cushion (mm) added to radial and |z| bounds to absorb
        numerical tails and rounding.

    Returns
    -------
    np.ndarray of bool
        True for hits compatible with at least one geometry element.
    """

    if layers is None or len(layers) == 0:
        # No geometry information – keep everything.
        return np.ones_like(r_vals, dtype=bool)

    r = np.asarray(r_vals, dtype=float)
    z = np.asarray(z_vals, dtype=float)
    abs_z = np.abs(z)
    mask = np.zeros_like(r, dtype=bool)

    tol = max(float(tolerance), 0.0)

    for layer in layers:
        layer_type = layer.get('type')

        if layer_type == 'disk':
            r_inner = layer.get('r_inner')
            r_outer = layer.get('r_outer')
            z_min = layer.get('z_min')
            z_max = layer.get('z_max')

            if r_inner is None or r_outer is None or z_min is None or z_max is None:
                continue

            r_lo = max(r_inner - tol, 0.0)
            r_hi = r_outer + tol
            z_lo = max(z_min - tol, 0.0)
            z_hi = z_max + tol

            layer_mask = (r >= r_lo) & (r <= r_hi) & (abs_z >= z_lo) & (abs_z <= z_hi)
            mask |= layer_mask
            continue

        # Treat anything else as a cylindrical section by default.
        r_inner = layer.get('r_inner')
        r_outer = layer.get('r_outer')
        radius = layer.get('radius')

        if r_inner is None and radius is not None:
            r_inner = radius
        if r_outer is None and radius is not None:
            r_outer = radius

        if r_inner is None and r_outer is None:
            # Cannot constrain radially; skip this layer.
            continue

        if r_outer is None:
            r_outer = r_inner
        if r_inner is None:
            r_inner = r_outer

        r_lo = max(min(r_inner, r_outer) - tol, 0.0)
        r_hi = max(r_inner, r_outer) + tol

        z_min = layer.get('z_min')
        z_max = layer.get('z_max')

        if z_min is not None and z_max is not None:
            z_lo = max(min(z_min, z_max) - tol, 0.0)
            z_hi = max(z_min, z_max) + tol
            z_mask = (abs_z >= z_lo) & (abs_z <= z_hi)
        else:
            # Many geometries omit explicit z extents; default to keeping hits.
            z_mask = np.ones_like(mask, dtype=bool)

        layer_mask = (r >= r_lo) & (r <= r_hi) & z_mask
        mask |= layer_mask

    return mask


def compute_rphi_area_map(layers: List[Dict], phi_edges: np.ndarray, r_edges: np.ndarray) -> np.ndarray:
    """Area (mm²) covered by each (phi, r) bin."""

    areas = np.zeros((len(phi_edges) - 1, len(r_edges) - 1), dtype=float)

    for iphi in range(len(phi_edges) - 1):
        dphi = abs(phi_edges[iphi + 1] - phi_edges[iphi])
        if dphi == 0:
            continue

        for ir in range(len(r_edges) - 1):
            r_lo = r_edges[ir]
            r_hi = r_edges[ir + 1]
            bin_area = 0.0

            for layer in layers:
                if layer['type'] == 'cylinder':
                    radius = layer.get('radius')
                    if radius is None:
                        r_inner = layer.get('r_inner')
                        r_outer = layer.get('r_outer')
                        if r_inner is not None and r_outer is not None:
                            radius = 0.5 * (r_inner + r_outer)
                    if radius is None:
                        continue
                    if (radius >= r_lo and radius < r_hi) or (ir == len(r_edges) - 2 and abs(radius - r_hi) < 1e-6):
                        length = layer.get('length')
                        if length is None:
                            z_min = layer.get('z_min')
                            z_max = layer.get('z_max')
                            if z_min is not None and z_max is not None:
                                length = abs(z_max - z_min)
                        if length:
                            bin_area += radius * dphi * length
                else:  # disk
                    r_inner = layer['r_inner']
                    r_outer = layer['r_outer']
                    overlap_lo = max(r_lo, r_inner)
                    overlap_hi = min(r_hi, r_outer)
                    if overlap_hi <= overlap_lo:
                        continue
                    bin_area += 0.5 * dphi * (overlap_hi ** 2 - overlap_lo ** 2)

            if bin_area <= 0.0:
                # Fallback wedge estimate
                bin_area = 0.5 * dphi * (r_hi ** 2 - r_lo ** 2)
            areas[iphi, ir] = bin_area

    return areas


def _z_overlap(z_lo: float, z_hi: float, z_min: Optional[float], z_max: Optional[float]) -> float:
    if z_min is None or z_max is None:
        return z_hi - z_lo
    return max(0.0, min(z_max, z_hi) - max(z_min, z_lo))


def compute_rz_area_map(layers: List[Dict], z_edges: np.ndarray, r_edges: np.ndarray) -> np.ndarray:
    """Area (mm²) covered by each (z, r) bin."""

    areas = np.zeros((len(z_edges) - 1, len(r_edges) - 1), dtype=float)

    for iz in range(len(z_edges) - 1):
        z_lo = z_edges[iz]
        z_hi = z_edges[iz + 1]
        dz_bin = abs(z_hi - z_lo)
        if dz_bin == 0:
            continue

        for ir in range(len(r_edges) - 1):
            r_lo = r_edges[ir]
            r_hi = r_edges[ir + 1]
            bin_area = 0.0

            for layer in layers:
                if layer['type'] == 'cylinder':
                    radius = layer.get('radius')
                    if radius is None:
                        r_inner = layer.get('r_inner')
                        r_outer = layer.get('r_outer')
                        if r_inner is not None and r_outer is not None:
                            radius = 0.5 * (r_inner + r_outer)
                    if radius is None:
                        continue

                    if (radius >= r_lo and radius < r_hi) or (ir == len(r_edges) - 2 and abs(radius - r_hi) < 1e-6):
                        overlap_z = _z_overlap(z_lo, z_hi, layer.get('z_min'), layer.get('z_max'))
                        if overlap_z > 0:
                            bin_area += 2.0 * np.pi * radius * overlap_z
                else:  # disk
                    r_inner = layer['r_inner']
                    r_outer = layer['r_outer']
                    overlap_lo = max(r_lo, r_inner)
                    overlap_hi = min(r_hi, r_outer)
                    if overlap_hi <= overlap_lo:
                        continue

                    z_min = layer.get('z_min')
                    z_max = layer.get('z_max')
                    overlap_z = _z_overlap(z_lo, z_hi, z_min, z_max)
                    if overlap_z <= 0:
                        continue

                    thickness = layer.get('thickness')
                    if thickness is None and z_min is not None and z_max is not None:
                        thickness = abs(z_max - z_min)
                    if thickness and thickness > 0:
                        fraction = min(1.0, overlap_z / thickness)
                    else:
                        fraction = 1.0

                    ring_area = np.pi * (overlap_hi ** 2 - overlap_lo ** 2)
                    bin_area += ring_area * fraction

            if bin_area <= 0.0:
                r_center = max(0.5 * (r_lo + r_hi), 1e-6)
                bin_area = 2.0 * np.pi * r_center * dz_bin

            areas[iz, ir] = bin_area

    return areas

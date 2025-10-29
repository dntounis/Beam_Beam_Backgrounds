"""
Helpers for computing sensitive detector areas directly from DD4hep geometry.

This module is intentionally standalone so it can be imported by external
analysis scripts without modifying the existing framework logic.
"""

import contextlib
import io
import json
import math
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

from src.detector_config import get_detector_configs
from src.geometry_parsing.geometry_info import get_geometry_info
from src.geometry_parsing.k4geo_parsers import parse_detector_constants, parse_value


Number = Optional[float]


def _to_float(value: Optional[float]) -> float:
    if value is None:
        raise ValueError("Required numeric value is missing while computing detector areas.")
    return float(value)


def _eval_attr(text: Optional[str], constants: Mapping[str, float]) -> Optional[float]:
    if text is None:
        return None
    result = parse_value(text, constants)
    if result is not None:
        return float(result)
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _regular_polygon_perimeter(num_sides: Optional[int], apothem: float) -> float:
    if num_sides is None or num_sides < 3:
        return 2.0 * math.pi * apothem
    return 2.0 * num_sides * apothem * math.tan(math.pi / num_sides)


def _regular_polygon_area(num_sides: Optional[int], apothem: float) -> float:
    if num_sides is None or num_sides < 3:
        return math.pi * apothem * apothem
    return num_sides * apothem * apothem * math.tan(math.pi / num_sides)


def _regular_polygon_ring_area(num_sides: Optional[int], inner: float, outer: float) -> float:
    return _regular_polygon_area(num_sides, outer) - _regular_polygon_area(num_sides, inner)


def _extract_numsides(root: ET.Element, constants: Mapping[str, float]) -> Optional[int]:
    shape = root.find(".//envelope/shape")
    if shape is None:
        return None
    num_sides = shape.get("numsides")
    value = _eval_attr(num_sides, constants)
    if value is None:
        return None
    return max(int(round(value)), 0)


def _detector_reflect_multiplier(detector_elem: ET.Element) -> int:
    reflect = detector_elem.get("reflect", "").strip().lower()
    return 2 if reflect in {"true", "1", "yes"} else 1


def _quiet_geometry_info(xml_path: str, config, constants: Mapping[str, float], main_xml: str) -> Mapping:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        info = get_geometry_info(xml_path, config, constants=constants, main_xml=main_xml)
    return info


def _extract_shape_dimensions(detector_elem: ET.Element, constants: Mapping[str, float]) -> Dict[str, Optional[float]]:
    shape = detector_elem.find(".//envelope/shape")
    if shape is None:
        return {"rmin": None, "rmax": None, "dz": None}
    dims = {}
    for key in ("rmin", "rmax", "dz", "inner_r", "outer_r"):
        dims[key] = _eval_attr(shape.get(key), constants)
    return dims


def _compute_barrel_tracker_area(geometry_info: Mapping) -> float:
    area = 0.0
    for layer in geometry_info.get("layers", {}).values():
        width = layer.get("width")
        length = layer.get("length")
        total_modules = layer.get("total_modules")
        total_pixels = layer.get("total_pixels")
        total_cells = layer.get("total_cells")

        if not all(v and v > 0 for v in (width, length, total_modules, total_pixels)):
            continue

        repeat = max(total_cells / (total_modules * total_pixels), 1.0)
        area_per_module = width * length
        area += area_per_module * total_modules * repeat
    return area


def _compute_endcap_tracker_area(geometry_info: Mapping) -> float:
    area = 0.0
    for layer in geometry_info.get("layers", {}).values():
        for ring in layer.get("rings", []):
            nmodules = ring.get("nmodules")
            total_pixels = ring.get("total_pixels")
            total_cells = ring.get("total_cells")

            if not nmodules or not total_pixels:
                continue

            repeat = max(total_cells / (nmodules * total_pixels), 1.0)

            if ring.get("type") == "trd":
                # Check if we have the new orientation-aware dimensions
                if "azimuthal_width_inner" in ring and "azimuthal_width_outer" in ring and "radial_thickness" in ring:
                    # New SiD-style orientation: use azimuthal width × radial thickness
                    azimuthal_width_avg = 0.5 * (ring["azimuthal_width_inner"] + ring["azimuthal_width_outer"])
                    radial_thickness = ring["radial_thickness"]
                    if azimuthal_width_avg > 0 and radial_thickness > 0:
                        area_module = azimuthal_width_avg * radial_thickness
                    else:
                        continue
                else:
                    # Fallback to historical calculation
                    x1 = ring.get("x1")
                    x2 = ring.get("x2")
                    z_len = ring.get("z")
                    if not all(v is not None and v > 0 for v in (x1, x2, z_len)):
                        continue
                    area_module = 0.5 * (x1 + x2) * z_len
            else:
                width = ring.get("width")
                length = ring.get("length")
                if not all(v is not None and v > 0 for v in (width, length)):
                    continue
                area_module = width * length

            area += area_module * nmodules * repeat
    return area


def _parse_layer_slices(layer_elem: ET.Element, constants: Mapping[str, float]) -> List[Tuple[float, bool]]:
    slices: List[Tuple[float, bool]] = []
    for slice_elem in layer_elem.findall("slice"):
        thickness = _eval_attr(slice_elem.get("thickness"), constants) or 0.0
        sensitive = slice_elem.get("sensitive", "").strip().lower() == "yes"
        slices.append((thickness, sensitive))
    return slices


def _compute_barrel_calo_area(
    geometry_info: Mapping,
    detector_elem: ET.Element,
    constants: Mapping[str, float],
) -> float:
    num_sides = _extract_numsides(detector_elem, constants)
    length = geometry_info.get("z_length") or geometry_info.get("length")
    inner_radius = geometry_info.get("rmin") or geometry_info.get("inner_r")
    if length is None or inner_radius is None:
        dims = _extract_shape_dimensions(detector_elem, constants)
        if inner_radius is None:
            inner_radius = dims.get("rmin") or dims.get("inner_r")
        if length is None:
            length = dims.get("dz")
    if length is None or inner_radius is None:
        raise ValueError("Missing barrel calorimeter dimensions.")

    radius = float(inner_radius)
    total_area = 0.0
    for layer_elem in detector_elem.findall(".//layer"):
        repeat = int(_eval_attr(layer_elem.get("repeat"), constants) or 1)
        slices = _parse_layer_slices(layer_elem, constants)
        if not slices:
            continue
        for _ in range(repeat):
            for thickness, sensitive in slices:
                if thickness <= 0:
                    radius += thickness
                    continue
                if sensitive:
                    mid_radius = radius + thickness * 0.5
                    perimeter = _regular_polygon_perimeter(num_sides, mid_radius)
                    total_area += perimeter * length
                radius += thickness
    return total_area


def _compute_endcap_calo_area(
    geometry_info: Mapping,
    detector_elem: ET.Element,
    constants: Mapping[str, float],
) -> float:
    num_sides = _extract_numsides(detector_elem, constants)
    inner = geometry_info.get("rmin") or geometry_info.get("inner_r")
    outer = geometry_info.get("rmax") or geometry_info.get("outer_r")
    if inner is None or outer is None:
        dims = _extract_shape_dimensions(detector_elem, constants)
        if inner is None:
            inner = dims.get("rmin") or dims.get("inner_r")
        if outer is None:
            outer = dims.get("rmax") or dims.get("outer_r")
    if inner is None or outer is None:
        raise ValueError("Missing endcap calorimeter radii.")

    base_area = _regular_polygon_ring_area(num_sides, inner, outer)
    total_area = 0.0
    for layer_elem in detector_elem.findall(".//layer"):
        repeat = int(_eval_attr(layer_elem.get("repeat"), constants) or 1)
        slices = _parse_layer_slices(layer_elem, constants)
        if not slices:
            continue
        sensitive_count = sum(1 for _, sensitive in slices if sensitive)
        total_area += base_area * sensitive_count * repeat
    return total_area


def _compute_forward_calo_area(
    geometry_info: Mapping,
    detector_elem: ET.Element,
    constants: Mapping[str, float],
) -> float:
    inner = geometry_info.get("inner_r") or geometry_info.get("rmin")
    outer = geometry_info.get("outer_r") or geometry_info.get("rmax")
    if inner is None or outer is None:
        dims = _extract_shape_dimensions(detector_elem, constants)
        if inner is None:
            inner = dims.get("inner_r") or dims.get("rmin")
        if outer is None:
            outer = dims.get("outer_r") or dims.get("rmax")
    if inner is None or outer is None:
        raise ValueError("Missing forward calorimeter radii.")

    area_per_disk = math.pi * (outer * outer - inner * inner)
    total_area = 0.0
    for layer_elem in detector_elem.findall(".//layer"):
        repeat = int(_eval_attr(layer_elem.get("repeat"), constants) or 1)
        slices = _parse_layer_slices(layer_elem, constants)
        if not slices:
            continue
        sensitive_count = sum(1 for _, sensitive in slices if sensitive)
        total_area += area_per_disk * sensitive_count * repeat
    return total_area


def compute_detector_areas(
    main_xml: str,
    detector_xmls: Mapping[str, str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute sensitive areas for the provided detectors.

    Parameters
    ----------
    main_xml : str
        Path to the main compact XML so constants can be resolved.
    detector_xmls : Mapping[str, str]
        Map of detector names to their specific XML files.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Per-detector info keyed by detector name. Each entry contains the
        sensitive area in mm^2 and cm^2 along with metadata.
    """

    configs = get_detector_configs()
    results: Dict[str, Dict[str, float]] = {}

    for detector, xml_path in detector_xmls.items():
        if detector not in configs:
            continue
        config = configs[detector]
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"XML file for {detector} not found: {xml_path}")

        constants = parse_detector_constants(main_xml, detector)
        geometry_info = _quiet_geometry_info(xml_path, config, constants=constants, main_xml=main_xml)

        tree = ET.parse(xml_path)
        detector_elem = tree.find(".//detector")
        if detector_elem is None:
            raise ValueError(f"Detector element missing in {xml_path}")
        reflect = _detector_reflect_multiplier(detector_elem)

        try:
            if config.detector_class in {"vertex", "tracker"}:
                if config.detector_type == "barrel":
                    area_mm2 = _compute_barrel_tracker_area(geometry_info)
                else:
                    area_mm2 = _compute_endcap_tracker_area(geometry_info)
            elif config.detector_class in {"ecal", "hcal", "muon"}:
                if config.detector_type == "barrel":
                    area_mm2 = _compute_barrel_calo_area(geometry_info, detector_elem, constants)
                else:
                    area_mm2 = _compute_endcap_calo_area(geometry_info, detector_elem, constants)
            elif config.detector_class in {"beamcal", "lumical"}:
                area_mm2 = _compute_forward_calo_area(geometry_info, detector_elem, constants)
            else:
                raise ValueError(f"Unsupported detector class: {config.detector_class}")
        except ValueError as exc:
            raise ValueError(f"Failed to compute area for {detector}: {exc}") from exc

        area_mm2 *= reflect
        area_cm2 = area_mm2 / 100.0

        results[detector] = {
            "area_mm2": area_mm2,
            "area_cm2": area_cm2,
            "reflect_multiplier": reflect,
            "detector_type": config.detector_type,
            "detector_class": config.detector_class,
        }

    return results


def save_area_report(
    report_path: str,
    area_data: Mapping[str, Mapping[str, float]],
    assumptions: Optional[Sequence[str]] = None,
) -> None:
    payload = {
        "detectors": area_data,
        "assumptions": list(assumptions or []),
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

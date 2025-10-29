import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex, to_rgb
import numpy as np

try:
    import mplhep as _mplhep  # type: ignore
    _HAS_MPLHEP = True
except Exception:
    _HAS_MPLHEP = False


SCENARIOS: Tuple[str, ...] = ("BL", "s.u.", "high-L")
COLLIDERS: Tuple[str, ...] = ("C3-250", "C3-550")
PARAMETER_SETS: Tuple[Tuple[str, str], ...] = (
    ("C3-250", "BL"),
    ("C3-250", "s.u."),
    ("C3-250", "high-L"),
    ("C3-550", "BL"),
    ("C3-550", "s.u."),
    ("C3-550", "high-L"),
)
COLLIDER_LABELS_LATEX: Mapping[str, str] = {
    "C3-250": r"$\mathrm{C}^{3}$ 250",
    "C3-550": r"$\mathrm{C}^{3}$ 550",
}
SCENARIO_LABELS_LATEX: Mapping[str, str] = {
    "BL": r"$\mathrm{BL}$",
    "s.u.": r"$\mathrm{s.u.}$",
    "high-L": r"$\mathrm{high-L}$",
}
PARAMETER_LABELS: Mapping[Tuple[str, str], str] = {
    key: rf"{COLLIDER_LABELS_LATEX.get(key[0], key[0])} {SCENARIO_LABELS_LATEX.get(key[1], key[1])}"
    for key in PARAMETER_SETS
}


class HitRateEntry(NamedTuple):
    value: float
    uncertainty: float


HitRateTable = Dict[str, Dict[str, Dict[str, HitRateEntry]]]
ComponentHitRateTable = Dict[str, HitRateTable]

COMPONENTS: Tuple[str, ...] = ("HPP", "IPC")
COMPONENT_COLOR_BLEND: Mapping[str, float] = {"HPP": 0.35, "IPC": 0.0}
COMPONENT_ALPHA: Mapping[str, float] = {"HPP": 0.85, "IPC": 1.0}
COMPONENT_LABELS: Mapping[str, str] = {
    "HPP": "HPP",
    "IPC": "IPC",
}


def _blend_with_white(color: str, factor: float) -> str:
    """Return color blended towards white by the given factor (0=no change, 1=white)."""
    factor = max(0.0, min(1.0, factor))
    r, g, b = to_rgb(color)
    blended = (
        (1.0 - factor) * r + factor,
        (1.0 - factor) * g + factor,
        (1.0 - factor) * b + factor,
    )
    return to_hex(blended)


def _component_color(base_color: str, component: str) -> str:
    blend_factor = COMPONENT_COLOR_BLEND.get(component, 0.0)
    return _blend_with_white(base_color, blend_factor)


DEFAULT_COMBINED_HIT_RATES: HitRateTable = {
    "barrel": {
        "Vertex": {
            "C3-250": {
                "BL": HitRateEntry(62.2, 1.2),
                "s.u.": HitRateEntry(124.2, 1.9),
                "high-L": HitRateEntry(158.2, 1.4),
            },
            "C3-550": {
                "BL": HitRateEntry(301.4, 7.9),
                "s.u.": HitRateEntry(597.2, 10.6),
                "high-L": HitRateEntry(600.8, 7.1),
            },
        },
        "Tracker": {
            "C3-250": {
                "BL": HitRateEntry(56.7, 0.6),
                "s.u.": HitRateEntry(112.6, 1.2),
                "high-L": HitRateEntry(142.9, 0.8),
            },
            "C3-550": {
                "BL": HitRateEntry(244.1, 2.8),
                "s.u.": HitRateEntry(490.7, 5.8),
                "high-L": HitRateEntry(498.8, 3.4),
            },
        },
        "ECAL": {
            "C3-250": {
                "BL": HitRateEntry(31.2, 0.5),
                "s.u.": HitRateEntry(62.5, 0.9),
                "high-L": HitRateEntry(82.5, 0.7),
            },
            "C3-550": {
                "BL": HitRateEntry(133.0, 1.7),
                "s.u.": HitRateEntry(266.3, 3.6),
                "high-L": HitRateEntry(280.5, 2.3),
            },
        },
        "HCAL": {
            "C3-250": {
                "BL": HitRateEntry(0.7, 0.1),
                "s.u.": HitRateEntry(1.6, 0.1),
                "high-L": HitRateEntry(3.3, 0.1),
            },
            "C3-550": {
                "BL": HitRateEntry(5.4, 0.3),
                "s.u.": HitRateEntry(11.0, 0.5),
                "high-L": HitRateEntry(16.9, 0.5),
            },
        },
        "Muon system": {
            "C3-250": {
                "BL": HitRateEntry(0.03, 0.01),
                "s.u.": HitRateEntry(0.06, 0.01),
                "high-L": HitRateEntry(0.12, 0.01),
            },
            "C3-550": {
                "BL": HitRateEntry(0.23, 0.03),
                "s.u.": HitRateEntry(0.41, 0.04),
                "high-L": HitRateEntry(0.50, 0.03),
            },
        },
    },
    "endcap": {
        "Vertex Endcap": {
            "C3-250": {
                "BL": HitRateEntry(34.4, 0.6),
                "s.u.": HitRateEntry(68.0, 1.0),
                "high-L": HitRateEntry(84.5, 0.8),
            },
            "C3-550": {
                "BL": HitRateEntry(202.1, 4.8),
                "s.u.": HitRateEntry(399.8, 8.2),
                "high-L": HitRateEntry(407.9, 4.5),
            },
        },
        "Vertex Forward": {
            "C3-250": {
                "BL": HitRateEntry(27.2, 0.5),
                "s.u.": HitRateEntry(54.4, 0.9),
                "high-L": HitRateEntry(67.7, 0.6),
            },
            "C3-550": {
                "BL": HitRateEntry(130.6, 2.5),
                "s.u.": HitRateEntry(260.8, 5.2),
                "high-L": HitRateEntry(268.1, 2.7),
            },
        },
        "Tracker": {
            "C3-250": {
                "BL": HitRateEntry(42.7, 0.7),
                "s.u.": HitRateEntry(85.2, 1.4),
                "high-L": HitRateEntry(108.4, 0.9),
            },
            "C3-550": {
                "BL": HitRateEntry(227.2, 4.3),
                "s.u.": HitRateEntry(455.4, 9.0),
                "high-L": HitRateEntry(467.8, 4.6),
            },
        },
        "ECAL": {
            "C3-250": {
                "BL": HitRateEntry(37.5, 1.0),
                "s.u.": HitRateEntry(74.8, 1.3),
                "high-L": HitRateEntry(95.7, 1.0),
            },
            "C3-550": {
                "BL": HitRateEntry(202.0, 4.5),
                "s.u.": HitRateEntry(401.9, 6.7),
                "high-L": HitRateEntry(430.5, 4.4),
            },
        },
        "HCAL": {
            "C3-250": {
                "BL": HitRateEntry(1220.0, 42.0),
                "s.u.": HitRateEntry(2436.0, 85.0),
                "high-L": HitRateEntry(3290.0, 69.0),
            },
            "C3-550": {
                "BL": HitRateEntry(6896.0, 240.0),
                "s.u.": HitRateEntry(13718.0, 484.0),
                "high-L": HitRateEntry(18488.0, 433.0),
            },
        },
        "Muon system": {
            "C3-250": {
                "BL": HitRateEntry(776.0, 20.0),
                "s.u.": HitRateEntry(1545.0, 41.0),
                "high-L": HitRateEntry(2149.0, 39.0),
            },
            "C3-550": {
                "BL": HitRateEntry(5082.0, 111.0),
                "s.u.": HitRateEntry(10117.0, 226.0),
                "high-L": HitRateEntry(12168.0, 195.0),
            },
        },
        "LumiCal": {
            "C3-250": {
                "BL": HitRateEntry(315.1, 4.2),
                "s.u.": HitRateEntry(628.8, 8.1),
                "high-L": HitRateEntry(772.9, 5.2),
            },
            "C3-550": {
                "BL": HitRateEntry(1790.3, 30.5),
                "s.u.": HitRateEntry(3574.6, 55.0),
                "high-L": HitRateEntry(3649.5, 28.9),
            },
        },
        "BeamCal": {
            "C3-250": {
                "BL": HitRateEntry(10139.0, 200.0),
                "s.u.": HitRateEntry(20177.0, 340.0),
                "high-L": HitRateEntry(21670.0, 206.0),
            },
            "C3-550": {
                "BL": HitRateEntry(84773.0, 2610.0),
                "s.u.": HitRateEntry(168471.0, 3750.0),
                "high-L": HitRateEntry(170871.0, 2389.0),
            },
        },
    },
}


def _blank_hit_rate_table(reference: HitRateTable) -> HitRateTable:
    return {
        region: {
            detector: {
                collider: {
                    scenario: HitRateEntry(0.0, 0.0)
                    for scenario in scenario_map
                }
                for collider, scenario_map in collider_map.items()
            }
            for detector, collider_map in detector_map.items()
        }
        for region, detector_map in reference.items()
    }


def _deep_copy_hit_rate_table(source: HitRateTable) -> HitRateTable:
    return {
        region: {
            detector: {
                collider: {
                    scenario: HitRateEntry(entry.value, entry.uncertainty)
                    for scenario, entry in scenario_map.items()
                }
                for collider, scenario_map in collider_map.items()
            }
            for detector, collider_map in detector_map.items()
        }
        for region, detector_map in source.items()
    }


DEFAULT_COMPONENT_HIT_RATES: ComponentHitRateTable = {
    "HPP": _blank_hit_rate_table(DEFAULT_COMBINED_HIT_RATES),
    "IPC": _deep_copy_hit_rate_table(DEFAULT_COMBINED_HIT_RATES),
}


def _combine_component_tables(component_tables: ComponentHitRateTable) -> HitRateTable:
    combined: HitRateTable = _blank_hit_rate_table(DEFAULT_COMBINED_HIT_RATES)
    for region, detector_map in combined.items():
        for detector, collider_map in detector_map.items():
            for collider, scenario_map in collider_map.items():
                for scenario in scenario_map:
                    total_value = 0.0
                    total_variance = 0.0
                    for component in COMPONENTS:
                        entry = component_tables[component][region][detector][collider][scenario]
                        total_value += entry.value
                        total_variance += entry.uncertainty ** 2
                    scenario_map[scenario] = HitRateEntry(total_value, float(np.sqrt(total_variance)))
    return combined


HIT_RATE_COMPONENTS: ComponentHitRateTable = {
    component: _deep_copy_hit_rate_table(DEFAULT_COMPONENT_HIT_RATES[component])
    for component in COMPONENTS
}
HIT_RATES: HitRateTable = _combine_component_tables(HIT_RATE_COMPONENTS)


def _initialize_hit_rates() -> None:
    _set_component_tables(DEFAULT_COMPONENT_HIT_RATES)


def _set_component_tables(component_tables: ComponentHitRateTable) -> None:
    global HIT_RATE_COMPONENTS, HIT_RATES
    missing = [component for component in COMPONENTS if component not in component_tables]
    if missing:
        raise KeyError(
            "Component tables missing required entries: {}".format(", ".join(sorted(missing)))
        )
    HIT_RATE_COMPONENTS = {
        component: _deep_copy_hit_rate_table(component_tables[component])
        for component in COMPONENTS
    }
    HIT_RATES = _combine_component_tables(HIT_RATE_COMPONENTS)


def _load_component_data(path: str) -> ComponentHitRateTable:
    with open(path, "r", encoding="utf-8") as handle:
        payload: Mapping[str, Any] = json.load(handle)

    component_tables: ComponentHitRateTable = {}
    for component in COMPONENTS:
        component_payload = payload.get(component)
        if component_payload is None:
            raise KeyError(f"Component '{component}' missing from JSON file '{path}'.")
        component_tables[component] = _parse_hit_rate_table(component_payload)
    return component_tables


def _parse_hit_rate_table(data: Mapping[str, Any]) -> HitRateTable:
    table: HitRateTable = {}
    for region, detector_map in data.items():
        region_lower = region.lower()
        table_region: Dict[str, Dict[str, Dict[str, HitRateEntry]]] = {}
        for detector, collider_map in detector_map.items():
            table_detector: Dict[str, Dict[str, HitRateEntry]] = {}
            for collider, scenario_map in collider_map.items():
                if collider not in COLLIDERS:
                    raise KeyError(f"Unknown collider '{collider}' in region '{region}'.")
                table_collider: Dict[str, HitRateEntry] = {}
                for scenario, entry_data in scenario_map.items():
                    if scenario not in SCENARIOS:
                        raise KeyError(
                            f"Unknown scenario '{scenario}' for collider '{collider}' in detector '{detector}'."
                        )
                    try:
                        value = float(entry_data["value"])
                        uncertainty = float(entry_data["uncertainty"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid entry for {region}/{detector}/{collider}/{scenario}: {entry_data}"
                        ) from exc
                    table_collider[scenario] = HitRateEntry(value, uncertainty)
                table_detector[collider] = table_collider
            table_region[detector] = table_detector
        table[region_lower] = table_region
    return table


SCENARIO_COLORS: Mapping[str, str] = {
    "BL": "#4c72b0",
    "s.u.": "#dd8452",
    "high-L": "#55a868",
}
COLLIDER_HATCHES: Mapping[str, str] = {
    "C3-250": "",
    "C3-550": "//",
}
BAR_EDGE_COLOR = "#1a1a1a"
BAR_LINEWIDTH = 0.7
ERROR_KW = {"elinewidth": 1.0, "capthick": 1.0, "capsize": 3.0}
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 13
TITLE_FONTSIZE = 16

_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_PROJECT_ROOT, ".."))

DEFAULT_GEOMETRY_DIR = os.path.join(
    _WORKSPACE_ROOT,
    "GuineaPig_July_2024",
    "k4geo",
    "SiD",
    "compact",
    "SiD_o2_v04",
)
DEFAULT_DD4HEP_PATH = os.path.join(_WORKSPACE_ROOT, "dd4hep_hit_analysis_framework")
DEFAULT_MAIN_XML = os.path.join(DEFAULT_GEOMETRY_DIR, "SiD_o2_v04.xml")

DETECTOR_XML_FILES: Mapping[str, str] = {
    "SiVertexBarrel": "SiVertexBarrel_o2_v04.xml",
    "SiVertexEndcap": "SiVertexEndcap_o2_v04.xml",
    "SiTrackerBarrel": "SiTrackerBarrel_o2_v04.xml",
    "SiTrackerEndcap": "SiTrackerEndcap_o2_v04.xml",
    "SiTrackerForward": "SiTrackerForward_o2_v04.xml",
    "ECalBarrel": "ECalBarrel_o2_v04.xml",
    "ECalEndcap": "ECalEndcap_o2_v04.xml",
    "HCalBarrel": "HCalBarrel_o2_v04.xml",
    "HCalEndcap": "HCalEndcap_o2_v04.xml",
    "MuonBarrel": "MuonBarrel_o2_v04.xml",
    "MuonEndcap": "MuonEndcap_o2_v04.xml",
    "BeamCal": "BeamCal_o2_v04.xml",
    "LumiCal": "LumiCal_o2_v04.xml",
}

REGION_LABEL_TO_DETECTOR: Mapping[str, Mapping[str, str]] = {
    "barrel": {
        "Vertex": "SiVertexBarrel",
        "Tracker": "SiTrackerBarrel",
        "ECAL": "ECalBarrel",
        "HCAL": "HCalBarrel",
        "Muon system": "MuonBarrel",
    },
    "endcap": {
        "Vertex Endcap": "SiVertexEndcap",
        "Vertex Forward": "SiTrackerForward",
        "Tracker": "SiTrackerEndcap",
        "ECAL": "ECalEndcap",
        "HCAL": "HCalEndcap",
        "Muon system": "MuonEndcap",
        "LumiCal": "LumiCal",
        "BeamCal": "BeamCal",
    },
}


def _apply_style() -> None:
    """Optionally activate mplhep CMS style."""
    if not _HAS_MPLHEP:
        return
    try:
        _mplhep.style.use("CMS")
    except Exception:
        pass


def _ensure_outdir(path: str) -> str:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def _region_arrays(region: str, table: HitRateTable) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Return detector labels and arrays indexed by parameter set."""
    detectors = list(table[region].keys())
    values = np.zeros((len(PARAMETER_SETS), len(detectors)), dtype=np.float64)
    errors = np.zeros_like(values)

    for det_idx, det in enumerate(detectors):
        for ps_idx, (collider, scenario) in enumerate(PARAMETER_SETS):
            entry = table[region][det][collider][scenario]
            values[ps_idx, det_idx] = entry.value
            errors[ps_idx, det_idx] = entry.uncertainty
    return detectors, values, errors


def _component_arrays(
    component: str, region: str, detectors: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    component_region = HIT_RATE_COMPONENTS[component][region]
    values = np.zeros((len(PARAMETER_SETS), len(detectors)), dtype=np.float64)
    errors = np.zeros_like(values)

    for det_idx, det in enumerate(detectors):
        detector_data = component_region.get(det)
        if detector_data is None:
            raise KeyError(f"Component '{component}' missing detector '{det}' in region '{region}'.")
        for ps_idx, (collider, scenario) in enumerate(PARAMETER_SETS):
            collider_data = detector_data.get(collider)
            if collider_data is None:
                raise KeyError(
                    f"Component '{component}' missing collider '{collider}' for detector '{det}' in region '{region}'."
                )
            entry = collider_data.get(scenario)
            if entry is None:
                raise KeyError(
                    "Component '{}' missing scenario '{}' for detector '{}' / collider '{}' in region '{}'".format(
                        component, scenario, det, collider, region
                    )
                )
            values[ps_idx, det_idx] = entry.value
            errors[ps_idx, det_idx] = entry.uncertainty
    return values, errors


def _normalize_by_area(
    detectors: Sequence[str],
    values: np.ndarray,
    errors: np.ndarray,
    area_lookup: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    norm_values = np.zeros_like(values, dtype=np.float64)
    norm_errors = np.zeros_like(errors, dtype=np.float64)
    for idx, det in enumerate(detectors):
        area = area_lookup.get(det)
        if area is None or area <= 0:
            raise ValueError(f"Missing or non-positive area for detector '{det}'.")
        norm_values[:, idx] = values[:, idx] / area
        norm_errors[:, idx] = errors[:, idx] / area
    return norm_values, norm_errors


def _print_density_summary(region: str, area_lookup: Mapping[str, float]) -> None:
    detectors, values, errors = _region_arrays(region, HIT_RATES)
    norm_values, norm_errors = _normalize_by_area(detectors, values, errors, area_lookup)
    print(f"{region.capitalize()} hit-rate densities (hits/ns/cm^2):")
    for det_idx, det in enumerate(detectors):
        print(f"  {det}:")
        for ps_idx, (collider, scenario) in enumerate(PARAMETER_SETS):
            label = f"{collider} {scenario}"
            value = norm_values[ps_idx, det_idx]
            err = norm_errors[ps_idx, det_idx]
            print(f"    {label:<12}: {value:.4g} ± {err:.2g}")


def _plot_bars(
    axes: Sequence[plt.Axes],
    detectors: Sequence[str],
    component_values: Mapping[str, np.ndarray],
    component_errors: Mapping[str, np.ndarray],
    *,
    ylabel: str,
    show_legend: bool,
    ymin: Optional[float] = None,
    legend_loc: Optional[str] = None,
) -> None:
    num_sets = len(PARAMETER_SETS)
    x_positions = np.arange(len(detectors), dtype=np.float64)
    bar_width = min(0.12, 0.8 / max(1, num_sets))
    offsets = (np.arange(num_sets) - (num_sets - 1) / 2.0) * bar_width

    for set_idx, (collider, scenario) in enumerate(PARAMETER_SETS):
        label = PARAMETER_LABELS[(collider, scenario)]
        color = SCENARIO_COLORS.get(scenario, "#4c72b0")
        hatch = COLLIDER_HATCHES.get(collider, "")
        for ax_idx, ax in enumerate(axes):
            bottom = np.zeros(len(detectors), dtype=np.float64)
            bar_label = label if show_legend and ax_idx == 0 else None
            for comp_idx, component in enumerate(COMPONENTS):
                comp_values = component_values[component][set_idx]
                comp_errors = component_errors[component][set_idx]
                comp_color = _component_color(color, component)
                alpha = COMPONENT_ALPHA.get(component, 1.0)

                # Show the collider/scenario label only on the top component to avoid duplicates
                comp_label = bar_label if (bar_label and comp_idx == len(COMPONENTS) - 1) else None
                ax.bar(
                    x_positions + offsets[set_idx],
                    comp_values,
                    width=bar_width,
                    bottom=bottom,
                    color=comp_color,
                    alpha=alpha,
                    edgecolor=BAR_EDGE_COLOR,
                    linewidth=BAR_LINEWIDTH,
                    hatch=hatch,
                    yerr=comp_errors,
                    error_kw=ERROR_KW,
                    label=comp_label,
                )
                bottom += comp_values

    for ax in axes:
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        ax.set_xlim(-0.5, len(detectors) - 0.5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(detectors, rotation=20, ha="right")
        #ax.set_xlabel("Subdetector", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, axis="y", linestyle="--", alpha=0.45)

    if show_legend and axes:
        legend_location = legend_loc
        if legend_location is None:
            # Choose legend location based on ylabel (hack: endcap uses "Endcap" in ylabel/title)
            legend_location = "upper left" if "Endcap" in ylabel or "endcap" in ylabel else "upper right"

        handles, labels = axes[0].get_legend_handles_labels()
        component_handles = [
            mpatches.Patch(
                facecolor=_component_color("#888888", component),
                alpha=COMPONENT_ALPHA.get(component, 1.0),
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_LINEWIDTH,
                label=COMPONENT_LABELS.get(component, component),
            )
            for component in COMPONENTS
        ]
        legend_main = axes[0].legend(
            handles,
            labels,
            loc=legend_location,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
        )

        legend_components = axes[0].legend(
            component_handles,
            [patch.get_label() for patch in component_handles],
            loc="center left",
            bbox_to_anchor=(0.3 if "Endcap" in ylabel or "endcap" in ylabel else 0.6, 0.75),
            bbox_transform=axes[0].transAxes,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
        )

        axes[0].add_artist(legend_main)
        axes[0].add_artist(legend_components)


def plot_barrel(outdir: str, formats: Iterable[str], dpi: int) -> None:
    detectors, values, errors = _region_arrays("barrel", HIT_RATES)
    component_arrays = {
        component: _component_arrays(component, "barrel", detectors)
        for component in COMPONENTS
    }
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    positive = values[values > 0]
    min_positive = float(np.min(positive)) if positive.size else 1e-3
    ymin = max(min_positive * 0.5, 1e-3)
    _plot_bars(
        [ax],
        detectors,
        {component: arrays[0] for component, arrays in component_arrays.items()},
        {component: arrays[1] for component, arrays in component_arrays.items()},
        ylabel="Hit rate (hits/ns)",
        show_legend=True,
        ymin=ymin,
        legend_loc="upper right",
    )

    ax.set_yscale("log")
    max_val = float(np.max(values + errors)) if values.size else ymin * 10.0
    ax.set_ylim(bottom=ymin, top=max_val * 1.3)
    ax.set_title("Background hit rates", fontsize=TITLE_FONTSIZE, loc="left")
    ax.set_title(r"$\mathbf{Barrel}$", fontsize=TITLE_FONTSIZE, loc="right")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))

    for ext in formats:
        out_path = os.path.join(outdir, f"hit_rates_barrel.{ext}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_barrel_density(
    outdir: str,
    formats: Iterable[str],
    dpi: int,
    area_lookup: Mapping[str, float],
) -> None:
    detectors, values, errors = _region_arrays("barrel", HIT_RATES)
    component_arrays = {
        component: _component_arrays(component, "barrel", detectors)
        for component in COMPONENTS
    }
    component_norm_arrays = {}
    for component, (comp_values, comp_errors) in component_arrays.items():
        norm_vals, norm_errs = _normalize_by_area(detectors, comp_values, comp_errors, area_lookup)
        component_norm_arrays[component] = (norm_vals, norm_errs)
    norm_values, norm_errors = _normalize_by_area(detectors, values, errors, area_lookup)
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    positive = norm_values[norm_values > 0]
    min_positive = float(np.min(positive)) if positive.size else 1e-6
    ymin = max(min_positive * 0.5, 1e-9)
    _plot_bars(
        [ax],
        detectors,
        {component: arrays[0] for component, arrays in component_norm_arrays.items()},
        {component: arrays[1] for component, arrays in component_norm_arrays.items()},
        ylabel=r"Hit rate density (hits/ns/cm$^2$)",
        show_legend=True,
        ymin=ymin,
        legend_loc="upper right",
    )
    ax.set_yscale("log")
    max_val = float(np.max(norm_values + norm_errors)) if norm_values.size else ymin * 10.0
    ax.set_ylim(bottom=ymin, top=max_val * 1.3)
    ax.set_title("Background hit-rate density", fontsize=TITLE_FONTSIZE, loc="left")
    ax.set_title(r"$\mathbf{Barrel}$", fontsize=TITLE_FONTSIZE, loc="right")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    for ext in formats:
        out_path = os.path.join(outdir, f"hit_rates_barrel_density.{ext}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_endcap(outdir: str, formats: Iterable[str], dpi: int) -> None:
    detectors, values, errors = _region_arrays("endcap", HIT_RATES)
    component_arrays = {
        component: _component_arrays(component, "endcap", detectors)
        for component in COMPONENTS
    }
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    positive = values[values > 0]
    min_positive = float(np.min(positive)) if positive.size else 1e-2
    ymin = max(min_positive * 0.5, 1e-2)
    _plot_bars(
        [ax],
        detectors,
        {component: arrays[0] for component, arrays in component_arrays.items()},
        {component: arrays[1] for component, arrays in component_arrays.items()},
        ylabel="Hit rate (hits/ns)",
        show_legend=True,
        ymin=ymin,
        legend_loc="upper left",
    )
    ax.set_yscale("log")
    max_val = float(np.max(values + errors)) if values.size else ymin * 10.0
    ax.set_ylim(bottom=ymin, top=max_val * 1.3)
    ax.set_title("Background hit rates", fontsize=TITLE_FONTSIZE, loc="left")
    ax.set_title(r"$\mathbf{Endcap}$", fontsize=TITLE_FONTSIZE, loc="right")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))

    for ext in formats:
        out_path = os.path.join(outdir, f"hit_rates_endcap.{ext}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_endcap_density(
    outdir: str,
    formats: Iterable[str],
    dpi: int,
    area_lookup: Mapping[str, float],
) -> None:
    detectors, values, errors = _region_arrays("endcap", HIT_RATES)
    component_arrays = {
        component: _component_arrays(component, "endcap", detectors)
        for component in COMPONENTS
    }
    component_norm_arrays = {}
    for component, (comp_values, comp_errors) in component_arrays.items():
        norm_vals, norm_errs = _normalize_by_area(detectors, comp_values, comp_errors, area_lookup)
        component_norm_arrays[component] = (norm_vals, norm_errs)
    norm_values, norm_errors = _normalize_by_area(detectors, values, errors, area_lookup)
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    positive = norm_values[norm_values > 0]
    min_positive = float(np.min(positive)) if positive.size else 1e-6
    ymin = max(min_positive * 0.5, 1e-6)
    _plot_bars(
        [ax],
        detectors,
        {component: arrays[0] for component, arrays in component_norm_arrays.items()},
        {component: arrays[1] for component, arrays in component_norm_arrays.items()},
        ylabel=r"Hit rate density (hits/ns/cm$^2$)",
        show_legend=True,
        ymin=ymin,
        legend_loc="upper center",
    )
    ax.set_yscale("log")
    max_val = float(np.max(norm_values + norm_errors)) if norm_values.size else ymin * 10.0
    ax.set_ylim(bottom=ymin, top=max_val * 1.3)
    ax.set_title("Background hit-rate density", fontsize=TITLE_FONTSIZE, loc="left")
    ax.set_title(r"$\mathbf{Endcap}$", fontsize=TITLE_FONTSIZE, loc="right")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    for ext in formats:
        out_path = os.path.join(outdir, f"hit_rates_endcap_density.{ext}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _parse_formats(fmt_list: Sequence[str]) -> List[str]:
    unique = []
    for item in fmt_list:
        if not item:
            continue
        value = item.lower().strip(".")
        if value and value not in unique:
            unique.append(value)
    return unique or ["png", "pdf"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot background hit-rate summary figures for barrel and endcap subdetectors."
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Output directory for generated plots (default: current directory)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "pdf"),
        help="Image formats to produce (default: png pdf)",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output resolution in DPI (default: 200)")
    parser.add_argument("--no-mplhep", action="store_true", help="Disable mplhep styling even if available")
    parser.add_argument(
        "--dd4hep-framework",
        default=DEFAULT_DD4HEP_PATH,
        help="Path to the dd4hep_hit_analysis_framework for area calculations.",
    )
    parser.add_argument(
        "--geometry-dir",
        default=DEFAULT_GEOMETRY_DIR,
        help="Directory containing the SiD_o2_v04 XML files (default: %(default)s).",
    )
    parser.add_argument(
        "--main-xml",
        default=DEFAULT_MAIN_XML,
        help="Path to the main compact XML file (default: %(default)s).",
    )
    parser.add_argument(
        "--area-report",
        default=None,
        help="Path for saving the computed detector-area JSON (default: <outdir>/detector_areas.json).",
    )
    parser.add_argument(
        "--skip-area-density",
        action="store_true",
        help="Only plot raw hit rates and skip the area-normalized figures.",
    )
    parser.add_argument(
        "--hit-rate-json",
        type=str,
        default=None,
        help=(
            "Optional path to a JSON file with separate component hit rates."
            " Must contain top-level keys 'HPP' and 'IPC'."
        ),
    )

    args = parser.parse_args()

    # Initialize data tables
    if args.hit_rate_json:
        _set_component_tables(_load_component_data(args.hit_rate_json))
    else:
        _initialize_hit_rates()

    if not args.no_mplhep:
        _apply_style()

    outdir = _ensure_outdir(args.outdir)
    formats = _parse_formats(args.formats)
    if not args.hit_rate_json:
        print("Using default hit-rate tables. Provide --hit-rate-json to override with new component data.")

    sys.path.append(args.dd4hep_framework)
    try:
        from src.utils.detector_area_helper import compute_detector_areas, save_area_report
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Unable to import detector_area_helper from dd4hep_hit_analysis_framework. "
            "Check --dd4hep-framework."
        ) from exc

    detector_xmls = {
        name: os.path.join(args.geometry_dir, filename) for name, filename in DETECTOR_XML_FILES.items()
    }
    area_data = compute_detector_areas(args.main_xml, detector_xmls)

    report_path = args.area_report or os.path.join(outdir, "detector_areas.json")
    assumptions = [
        "Silicon tracker modules are treated as fully sensitive across their envelopes.",
        "Calorimeter and muon sensitive slices are detected via sensitive='yes' markers in the XML.",
    ]
    save_area_report(report_path, area_data, assumptions)

    region_area_lookup: Dict[str, Dict[str, float]] = {"barrel": {}, "endcap": {}}
    for region, mapping in REGION_LABEL_TO_DETECTOR.items():
        for label, detector in mapping.items():
            detector_area = area_data.get(detector)
            if detector_area is None:
                raise KeyError(f"Area information for detector '{detector}' is missing.")
            region_area_lookup[region][label] = detector_area["area_cm2"]

    print("Detector areas (cm^2):")
    for region in ("barrel", "endcap"):
        for label, detector in REGION_LABEL_TO_DETECTOR[region].items():
            area_cm2 = region_area_lookup[region][label]
            print(f"  {region.capitalize():>6} | {label:<15} ({detector}): {area_cm2:10.3f} cm^2")
    print(f"Area report saved to {report_path}")
    _print_density_summary("barrel", region_area_lookup["barrel"])
    _print_density_summary("endcap", region_area_lookup["endcap"])

    plot_barrel(outdir, formats, args.dpi)
    plot_endcap(outdir, formats, args.dpi)

    if not args.skip_area_density:
        plot_barrel_density(outdir, formats, args.dpi, region_area_lookup["barrel"])
        plot_endcap_density(outdir, formats, args.dpi, region_area_lookup["endcap"])
        print("Generated area-normalized hit-rate figures.")

    print(f"Saved figures to {outdir}")


if __name__ == "__main__":
    main()

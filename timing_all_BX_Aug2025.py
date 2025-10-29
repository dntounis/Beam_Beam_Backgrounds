import os
import re
import argparse
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
from typing import Optional

try:
    from scipy.optimize import curve_fit
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


TRACKER_BRANCHES = [
    "SiVertexBarrelHits",
    "SiVertexEndcapHits",
    "SiTrackerEndcapHits",
    "SiTrackerBarrelHits",
    "SiTrackerForwardHits",
]

CALO_BRANCHES = [
    "ECalBarrelHits",
    "ECalEndcapHits",
    "HCalBarrelHits",
    "HCalEndcapHits",
    "MuonBarrelHits",
    "MuonEndcapHits",
    "LumiCalHits",
    "BeamCalHits",
]

ALL_BRANCHES = TRACKER_BRANCHES + CALO_BRANCHES

BRANCH_TO_RATE_CATEGORY = {
    "SiVertexBarrelHits": ("barrel", "Vertex"),
    "SiTrackerBarrelHits": ("barrel", "Tracker"),
    "ECalBarrelHits": ("barrel", "ECAL"),
    "HCalBarrelHits": ("barrel", "HCAL"),
    "MuonBarrelHits": ("barrel", "Muon system"),
    "SiVertexEndcapHits": ("endcap", "Vertex Endcap"),
    "SiTrackerForwardHits": ("endcap", "Vertex Forward"),
    "SiTrackerEndcapHits": ("endcap", "Tracker"),
    "ECalEndcapHits": ("endcap", "ECAL"),
    "HCalEndcapHits": ("endcap", "HCAL"),
    "MuonEndcapHits": ("endcap", "Muon system"),
    "LumiCalHits": ("endcap", "LumiCal"),
    "BeamCalHits": ("endcap", "BeamCal"),
}


def _parse_collider_label(label: str) -> tuple[str, str]:
    label = label.strip()
    if not label:
        return "unknown", "default"
    parts = label.split()
    if len(parts) == 1:
        return parts[0], "default"
    return parts[0], " ".join(parts[1:])


def _format_json_number(value: float) -> float | int:
    if not np.isfinite(value):  # type: ignore[arg-type]
        return 0.0
    rounded = float(round(value, 4))
    if abs(rounded - round(rounded)) < 1e-6:
        return int(round(rounded))
    return rounded


def _merge_nested_dict(target: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict):
            node = target.setdefault(key, {})
            if isinstance(node, dict):
                _merge_nested_dict(node, value)
            else:
                target[key] = value
        else:
            target[key] = value


def compute_rate_stats(
    counts: np.ndarray,
    edges: np.ndarray,
    time_min: float,
    time_max: float,
    plot_binwidth: float,
) -> tuple[tuple[float, float] | None, str | None]:
    if counts.size == 0:
        # Empty histogram but considered valid with zero rate
        if plot_binwidth <= 0:
            return None, "invalid_binwidth"
        return ((0.0, 0.0)), None
    try:
        counts_reb, edges_reb = rebin_counts(counts, edges, plot_binwidth, time_min, time_max)
    except ValueError:
        return None, "rebin_failed"
    centers = 0.5 * (edges_reb[:-1] + edges_reb[1:])
    fit_mask = (centers >= (edges[0] + 5.0)) & (centers <= (edges[-1] - 5.0))
    if not bool(fit_mask.size and fit_mask.any()):
        return None, "insufficient_bins"
    mean_bin, err_bin = _fit_const(centers[fit_mask], counts_reb[fit_mask])
    if plot_binwidth <= 0:
        return None, "invalid_binwidth"
    mean_per_ns = mean_bin / plot_binwidth
    err_per_ns = err_bin / plot_binwidth
    return (float(mean_per_ns), float(err_per_ns)), None


# SiVertexBarrel layer geometry (mm)
VB_LOWER_R = np.array([13.0, 21.0, 34.0, 46.6, 59.0], dtype=np.float64)
VB_UPPER_R = np.array([17.0, 25.0, 38.0, 50.6, 63.0], dtype=np.float64)
VB_RADIAL_CENTERS = np.array([15.05, 23.03, 35.79, 47.50, 59.90], dtype=np.float64)
VB_ACTIVE_LENGTH_MM = 126.0  # 2 * 63 mm


def discover_files(base_dir: str, pattern: str) -> list[tuple[int, str]]:
    """Return sorted list of (seed, filepath) discovered under base_dir.

    pattern should contain a single "*" where the seed number appears,
    e.g. "ddsim_C3_250_PS1_v2_seed_*.edm4hep.root".
    """
    full_pattern = os.path.join(base_dir, pattern)
    files = glob.glob(full_pattern)
    out = []
    # Support filenames with optional suffix after the seed, e.g. ...seed_123.edm4hep.root or ...seed_123_MERGED.edm4hep.root
    seed_re = re.compile(r"seed_(\d+).*\.edm4hep\.root$")
    for fp in files:
        m = seed_re.search(fp)
        if not m:
            continue
        out.append((int(m.group(1)), fp))
    out.sort(key=lambda x: x[0])
    return out


def _find_tracker_energy_leaf(tree: uproot.TTree, branch: str) -> str | None:
    """Return the tracker energy leaf name if present (expected: '<branch>.eDep')."""
    name = f"{branch}.eDep"
    try:
        # attempt to access; uproot will raise KeyError if missing
        _ = tree.keys(filter_name=name)
        # keys() with exact filter may still return empty list; verify by trying arrays request lazily
        return name
    except Exception:
        return None


def read_tracker_times(
    tree: uproot.TTree,
    branch: str,
    energy_threshold: float = 0.0,
    return_raw_count: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Read tracker hit times for all events in the file.

    Returns a 1D numpy array of times after optional energy filtering.
    When ``return_raw_count`` is True, also returns the number of hits before
    any filtering so raw hit totals can be compared against filtered counts.
    """
    names = [
        f"{branch}",
        f"{branch}.position.x",
        f"{branch}.position.y",
        f"{branch}.position.z",
        f"{branch}.time",
    ]
    energy_leaf = _find_tracker_energy_leaf(tree, branch)
    if energy_leaf is not None:
        names.append(energy_leaf)
    arrays = tree.arrays(names, library="ak")
    posx = arrays[f"{branch}.position.x"]
    posy = arrays[f"{branch}.position.y"]
    posz = arrays[f"{branch}.position.z"]
    time = arrays[f"{branch}.time"]
    if energy_leaf is not None:
        energy = arrays[energy_leaf]
    else:
        energy = None

    mask = ak.full_like(posx, True, dtype=np.bool_)
    #mask = ~((posx == 0) & (posy == 0) & (posz == 0))
    # mask = (posx != 0) & (posy != 0) & (posz != 0)
    if energy is not None and energy_threshold > 0.0:
        mask = mask & (energy >= energy_threshold)

    if len(time) == 0:
        raw_count = 0
        times = np.empty((0,), dtype=np.float64)
    else:
        raw_count = int(ak.sum(ak.num(time, axis=1)))
        filtered = ak.flatten(time[mask], axis=None)
        times = ak.to_numpy(filtered).astype(np.float64, copy=False)

    if return_raw_count:
        return times, raw_count
    return times


def read_tracker_xyz_time(
    tree: uproot.TTree,
    branch: str,
    energy_threshold: float = 0.0,
    return_raw_count: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Read tracker hit x,y,z,time arrays (filtered) for all events and optional raw count."""
    names = [
        f"{branch}",
        f"{branch}.position.x",
        f"{branch}.position.y",
        f"{branch}.position.z",
        f"{branch}.time",
    ]
    energy_leaf = _find_tracker_energy_leaf(tree, branch)
    if energy_leaf is not None:
        names.append(energy_leaf)
    arrays = tree.arrays(names, library="ak")
    posx = arrays[f"{branch}.position.x"]
    posy = arrays[f"{branch}.position.y"]
    posz = arrays[f"{branch}.position.z"]
    time = arrays[f"{branch}.time"]
    if energy_leaf is not None:
        energy = arrays[energy_leaf]
    else:
        energy = None
    # we filter out hits where ALL coordinates are zero
    valid = ak.full_like(posx, True, dtype=np.bool_)
    #valid = ~((posx == 0) & (posy == 0) & (posz == 0))
    # valid = (posx != 0) & (posy != 0) & (posz != 0)
    if energy is not None and energy_threshold > 0.0:
        valid = valid & (energy >= energy_threshold)

    if len(time) == 0:
        raw_count = 0
        has_valid = False
    else:
        raw_count = int(ak.sum(ak.num(time, axis=1)))
        has_valid = bool(ak.any(valid))

    if raw_count == 0 or not has_valid:
        x = np.empty((0,), dtype=np.float64)
        y = np.empty((0,), dtype=np.float64)
        z = np.empty((0,), dtype=np.float64)
        t = np.empty((0,), dtype=np.float64)
    else:
        x = ak.to_numpy(ak.flatten(posx[valid], axis=None)).astype(np.float64, copy=False)
        y = ak.to_numpy(ak.flatten(posy[valid], axis=None)).astype(np.float64, copy=False)
        z = ak.to_numpy(ak.flatten(posz[valid], axis=None)).astype(np.float64, copy=False)
        t = ak.to_numpy(ak.flatten(time[valid], axis=None)).astype(np.float64, copy=False)

    if return_raw_count:
        return x, y, z, t, raw_count
    return x, y, z, t


def read_calo_times(
    tree: uproot.TTree,
    branch: str,
    energy_threshold: float = 0.0,
    return_raw_count: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Read calorimeter hit times using the first contribution per hit.

    Uses:
      - {branch}.position.{x,y,z} to filter non-zero hit positions
      - {branch}.contributions_begin / contributions_end to pick contribution range
      - {branch}Contributions.time to get time; pick first contribution per hit
    Returns 1D numpy array of times.
    """
    names = [
        f"{branch}",
        f"{branch}.position.x",
        f"{branch}.position.y",
        f"{branch}.position.z",
        f"{branch}.contributions_begin",
        f"{branch}.contributions_end",
        f"{branch}Contributions.time",
    ]
    # energy: for calo contributions we expect '<branch>Contributions.energy'
    energy_leaf = f"{branch}Contributions.energy"
    all_keys = set(tree.keys())
    if energy_leaf in all_keys:
        names.append(energy_leaf)
    else:
        energy_leaf = None
    arrays = tree.arrays(names, library="ak")

    posx = arrays[f"{branch}.position.x"]
    posy = arrays[f"{branch}.position.y"]
    posz = arrays[f"{branch}.position.z"]
    beg = arrays[f"{branch}.contributions_begin"]
    end = arrays[f"{branch}.contributions_end"]
    ctime = arrays[f"{branch}Contributions.time"]
    if energy_leaf is not None:
        energy = arrays[energy_leaf]
    else:
        energy = None

    # valid hits: non-zero pos and has at least one contribution
    # we filter out hits where ALL coordinates are zero
    hit_valid = ak.full_like(posx, True, dtype=np.bool_)
    #hit_valid = ~((posx == 0) & (posy == 0) & (posz == 0)) & (beg < end)
    # hit_valid = (posx != 0) & (posy != 0) & (posz != 0) & (beg < end)
    hit_valid = hit_valid & (beg < end)
    # apply energy threshold at the first contribution per hit
    if energy is not None and energy_threshold > 0.0:
        first_energy = energy[beg]
        hit_valid = hit_valid & (first_energy >= energy_threshold)

    if len(posx) == 0:
        raw_count = 0
        valid_hits = False
    else:
        raw_count = int(ak.sum(ak.num(posx, axis=1)))
        valid_hits = bool(ak.any(hit_valid))

    if raw_count == 0 or not valid_hits:
        times = np.empty((0,), dtype=np.float64)
    else:
        # first contribution time per valid hit
        first_time = ctime[beg]
        times = ak.to_numpy(ak.flatten(first_time[hit_valid], axis=None)).astype(np.float64, copy=False)

    if return_raw_count:
        return times, raw_count
    return times


def make_bins(lo: float, hi: float, base_binwidth: float) -> np.ndarray:
    nbins = int(round((hi - lo) / base_binwidth))
    return np.linspace(lo, hi, nbins + 1, dtype=np.float64)


def rebin_counts(counts: np.ndarray, edges: np.ndarray, new_binwidth: float, xmin: float, xmax: float) -> tuple[np.ndarray, np.ndarray]:
    base_bw = edges[1] - edges[0]
    ratio = new_binwidth / base_bw
    if abs(round(ratio) - ratio) > 1e-6:
        raise ValueError(f"plot bin width {new_binwidth} is not a multiple of base bin width {base_bw}")
    ratio = int(round(ratio))
    # restrict to [xmin,xmax]
    i0 = max(0, int(np.floor((xmin - edges[0]) / base_bw)))
    i1 = min(counts.size, int(np.ceil((xmax - edges[0]) / base_bw)))
    counts_win = counts[i0:i1]
    # pad to multiple of ratio
    pad = (-counts_win.size) % ratio
    if pad:
        counts_win = np.pad(counts_win, (0, pad), mode='constant')
    rebinned = counts_win.reshape(-1, ratio).sum(axis=1)
    edges_new = np.arange(edges[0] + i0 * base_bw, edges[0] + (i0 + rebinned.size * ratio) * base_bw + 1e-12, new_binwidth)
    return rebinned, edges_new


def accumulate_hpp_counts(
    counts_per_bx: dict[str, np.ndarray],
    edges: np.ndarray,
    branches: list[str],
    bunch_spacing: float,
    display_bunches: int,
) -> dict[str, np.ndarray]:
    if edges.size < 2:
        return {b: np.zeros(0, dtype=np.float64) for b in branches}
    base_bw = edges[1] - edges[0]
    shift_bins = int(round(bunch_spacing / base_bw)) if base_bw > 0 else 0
    accum: dict[str, np.ndarray] = {b: np.zeros(edges.size - 1, dtype=np.float64) for b in branches}
    for b in branches:
        per_bx = counts_per_bx.get(b)
        if per_bx is None or per_bx.size == 0:
            continue
        total = accum[b]
        for ib in range(display_bunches):
            shift = ib * shift_bins
            if shift >= total.size:
                break
            end = min(shift + per_bx.size, total.size)
            span = end - shift
            if span > 0:
                total[shift:end] += per_bx[:span]
    return accum


def load_histogram_npz(npz_path: str) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, float],
    list[str],
    dict[str, np.ndarray],
]:
    with np.load(npz_path, allow_pickle=True) as npz:
        edges = npz["edges"]
        branches = ALL_BRANCHES
        counts_total: dict[str, np.ndarray] = {}
        counts_ipc: dict[str, np.ndarray] = {}
        counts_hpp: dict[str, np.ndarray] = {}
        raw_entries: dict[str, float] = {}
        base_len = max(edges.size - 1, 0)
        for b in branches:
            key_total = f"counts_{b}"
            if key_total in npz:
                counts_total[b] = npz[key_total]
            else:
                counts_total[b] = np.zeros(base_len, dtype=np.float64)
            key_ipc = f"counts_ipc_{b}"
            if key_ipc in npz:
                counts_ipc[b] = npz[key_ipc]
            else:
                counts_ipc[b] = counts_total[b].copy()
            key_hpp = f"counts_hpp_{b}"
            if key_hpp in npz:
                counts_hpp[b] = npz[key_hpp]
            else:
                counts_hpp[b] = counts_total[b] - counts_ipc[b]
            raw_entries[b] = float(npz.get(f"raw_entries_{b}", np.sum(counts_total[b])))
        cached_files = list(npz.get("files", []))
        hpp_counts_per_bx: dict[str, np.ndarray] = {}
        for b in ALL_BRANCHES:
            key = f"hpp_counts_per_bx_{b}"
            if key in npz:
                hpp_counts_per_bx[b] = npz[key]
        return edges, counts_total, counts_ipc, counts_hpp, raw_entries, cached_files, hpp_counts_per_bx


def _place_text_avoid_overlap(ax: plt.Axes, text: str, fontsize: int) -> None:
    """Place a text box in one of the corners, avoiding overlap with legend if possible.

    Tries TL, TR, BL, BR with a semi-transparent white background for readability.
    """
    fig = ax.figure
    # Ensure a renderer exists to compute extents
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        renderer = None

    legend = ax.get_legend()
    legend_bb = None
    if legend is not None and renderer is not None:
        try:
            legend_bb = legend.get_window_extent(renderer=renderer)
        except Exception:
            legend_bb = None

    candidates = [
        (0.02, 0.98, 'left', 'top'),
        (0.98, 0.98, 'right', 'top'),
        (0.02, 0.02, 'left', 'bottom'),
        (0.98, 0.02, 'right', 'bottom'),
    ]

    placed = None
    for x, y, ha, va in candidates:
        t = ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=fontsize,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2),
        )
        if renderer is None:
            placed = t
            break
        try:
            fig.canvas.draw()
            tb = t.get_window_extent(renderer=renderer)
            if legend_bb is not None and tb.overlaps(legend_bb):
                t.remove()
                continue
            placed = t
            break
        except Exception:
            placed = t
            break

    if placed is None:
        # Fallback to top-right
        ax.text(
            0.98,
            0.98,
            text,
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=fontsize,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2),
        )


def plot_hist(
    branch: str,
    counts: np.ndarray,
    edges: np.ndarray,
    outdir: str,
    plot_xmin: float,
    plot_xmax: float,
    plot_binwidth: float,
    collider: str,
    detector_version: str,
    n_bunches: int,
    use_mplhep: bool = True,
    title_fs: int = 15,
    label_fs: int = 14,
    tick_fs: int = 11,
    legend_fs: int = 13,
    variant_label: Optional[str] = None,
    filename_suffix: Optional[str] = None,
) -> tuple[str, str]:
    try:
        if use_mplhep:
            import mplhep as hep  # noqa: F401
            # stylistics only
            import mplhep as hep_style
            hep_style.style.use("CMS")
    except Exception:
        pass

    # Rebin for plotting
    counts_reb, edges_reb = rebin_counts(counts, edges, plot_binwidth, plot_xmin, plot_xmax)
    bin_width = plot_binwidth
    centers = 0.5 * (edges_reb[:-1] + edges_reb[1:])

    # Compute constant fit (mean) excluding first/last 5 ns of the FULL time span
    # Full span approximated by n_bunches * bunch spacing which we embed as plot_xmax if larger
    full_min = edges[0]
    full_max = edges[-1]
    mask = (centers >= (full_min + 5.0)) & (centers <= (full_max - 5.0))
    y_fit = counts_reb[mask]
    if _HAS_SCIPY and y_fit.size > 0:
        def _const(x, a):
            return np.full_like(x, a)
        x_fit = centers[mask]
        try:
            popt, pcov = curve_fit(_const, x_fit, y_fit)
            perr = np.sqrt(np.diag(pcov)) if pcov.size else np.array([0.0])
            mean = float(popt[0])
            stderr = float(perr[0])
        except Exception:
            mean = float(np.mean(y_fit))
            stderr = float(np.std(y_fit, ddof=1) / np.sqrt(max(1, y_fit.size)))
    else:
        if y_fit.size > 1:
            mean = float(np.mean(y_fit))
            stderr = float(np.std(y_fit, ddof=1) / np.sqrt(y_fit.size))
        else:
            mean = float(np.mean(y_fit) if y_fit.size else 0.0)
            stderr = 0.0

    # Plot
    plt.figure(figsize=(8, 5))
    plt.step(centers, counts_reb / bin_width, where="mid", color='blue', label='Data')
    plt.plot(centers, np.full_like(centers, mean) / bin_width, color='red', label='Constant fit')
    plt.fill_between(
        centers,
        (mean - stderr) / bin_width,
        (mean + stderr) / bin_width,
        color='gray', alpha=0.4, label='68% CL'
    )
    plt.xlim(plot_xmin, plot_xmax)
    ax = plt.gca()
    ax.tick_params(labelsize=tick_fs)
    plt.xlabel('Time (ns)', fontsize=label_fs)
    plt.ylabel(f'Number of hits/{bin_width:.2f} ns', fontsize=label_fs)
    # Title left: collider and bunches; right: detector version and subdetector
    bunch_text = f"{n_bunches} bunches" if n_bunches != 1 else "1 bunch"
    left_title = f"{collider} ({bunch_text})"
    right_title = f"{detector_version} - {branch}"
    if variant_label:
        right_title = f"{right_title} ({variant_label})"
    plt.title(left_title, fontsize=title_fs, loc='left')
    plt.title(right_title, fontsize=title_fs, loc='right')
    text_fs=label_fs+2
    plt.legend(fontsize=legend_fs)
    # Place mean text after legend is drawn, avoiding overlap
    _place_text_avoid_overlap(
        ax,
        f'mean = {mean/bin_width:.2f} ± {stderr/bin_width:.2f} hits/ns',
        fontsize=text_fs,
    )
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    png = os.path.join(outdir, f"{collider}_timing_{branch}{suffix}.png")
    pdf = os.path.join(outdir, f"{collider}_timing_{branch}{suffix}.pdf")
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    return png, pdf


def _const(x, a):
    return np.full_like(x, a)


def _fit_const(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size == 0 or y.size == 0:
        return 0.0, 0.0
    if _HAS_SCIPY:
        try:
            popt, pcov = curve_fit(_const, x, y)
            perr = np.sqrt(np.diag(pcov)) if pcov.size else np.array([0.0])
            return float(popt[0]), float(perr[0])
        except Exception:
            pass
    mean = float(np.mean(y))
    err = float(np.std(y, ddof=1) / np.sqrt(max(1, y.size))) if y.size > 1 else 0.0
    return mean, err


def plot_tracker_overlay(hist_counts: dict, edges: np.ndarray, outdir: str,
                         branches: Optional[list] = None,
                         time_min: float = 0.0, time_max: float = 800.0,
                         plot_binwidth: float = 10.0,
                         y_max: Optional[float] = None,
                         use_mplhep: bool = True,
                         legend_fs: int = 10) -> None:
    if branches is None:
        branches = [
            "SiVertexBarrelHits",
            "SiTrackerBarrelHits",
            "SiTrackerEndcapHits",
            "SiTrackerForwardHits",
            "SiVertexEndcapHits",
        ]
    try:
        if use_mplhep:
            import mplhep as hep_style
            hep_style.style.use("CMS")
    except Exception:
        pass

    colors = ['blue', 'red', 'green', 'purple', 'orange']
    plt.figure(figsize=(8, 5))
    for i, b in enumerate(branches):
        if b not in hist_counts:
            continue
        counts = hist_counts[b]
        c_reb, e_reb = rebin_counts(counts, edges, plot_binwidth, time_min, time_max)
        centers = 0.5 * (e_reb[:-1] + e_reb[1:])
        mask = (centers >= (time_min + 5.0)) & (centers <= (time_max - 5.0))
        mean, err = _fit_const(centers[mask], c_reb[mask])
        plt.hist(centers, bins=e_reb, weights=c_reb/plot_binwidth, histtype='step',
                 color=colors[i % len(colors)], label=b)
        plt.plot(centers, np.full_like(centers, mean)/plot_binwidth,
                 color=colors[i % len(colors)], linestyle='dashed')
        plt.fill_between(centers, (mean-err)/plot_binwidth, (mean+err)/plot_binwidth,
                         color=colors[i % len(colors)], alpha=0.4)
    plt.xlabel('Time (ns)')
    plt.ylabel(f'Number of hits/{plot_binwidth:.2f} ns')
    plt.xlim(time_min, time_max)
    if y_max is not None:
        plt.ylim(0, y_max)
    plt.legend(loc='upper right', fontsize=legend_fs)
    plt.savefig(os.path.join(outdir, 'Timing_All.png'), bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'Timing_All.pdf'), bbox_inches='tight')
    plt.close()


def plot_vertex_barrel_layers_from_root(root_path: str, outdir: str,
                                        time_min: float, time_max: float,
                                        plot_binwidth: float,
                                        y_max: Optional[float] = None,
                                        use_mplhep: bool = True) -> None:
    try:
        if use_mplhep:
            import mplhep as hep_style
            hep_style.style.use("CMS")
    except Exception:
        pass

    hist_names = [
        "h_timing_all_layers",
        "h_timing_1st_layer",
        "h_timing_2nd_layer",
        "h_timing_3rd_layer",
        "h_timing_4th_layer",
        "h_timing_5th_layer",
    ]
    hist_names_area = [
        "h_timing_1st_layer",
        "h_timing_2nd_layer",
        "h_timing_3rd_layer",
        "h_timing_4th_layer",
        "h_timing_5th_layer",
    ]
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    plt_names = ['all layers', '1st layer', '2nd layer', '3rd layer', '4th layer', '5th layer']
    radius_text = [' ', 'r=15.05 mm', 'r=23.03 mm', 'r=35.79 mm', 'r=47.50 mm', 'r=59.90 mm']
    radii = [15.05, 23.03, 35.79, 47.50, 59.90]
    length_mm = 126.0  # 2*63 mm
    areas = [2*np.pi*r*length_mm for r in radii]  # mm^2

    with uproot.open(root_path) as f:
        # Figure 1: hits/ns, Figure 2: hits
        fig1, ax1 = plt.subplots(figsize=(8,5))
        fig2, ax2 = plt.subplots(figsize=(8,5))
        for i, hname in enumerate(hist_names):
            if hname not in f:
                continue
            counts, edges = f[hname].to_numpy()
            # rebin to requested plot_binwidth
            base_bw = edges[1] - edges[0]
            ratio = int(round(plot_binwidth / base_bw))
            counts = counts.reshape(-1, ratio).sum(axis=1)
            edges = edges[::ratio]
            x = 0.5*(edges[:-1]+edges[1:])
            mask = (x >= time_min+5) & (x <= time_max-5)
            mean, err = _fit_const(x[mask], counts[mask])
            ax1.hist(x, bins=edges, weights=counts/plot_binwidth, histtype='step',
                     color=colors[i%len(colors)], label=f'{plt_names[i]} {radius_text[i]}')
            ax1.plot(x, np.full_like(x, mean)/plot_binwidth, color=colors[i%len(colors)], linestyle='dashed')
            ax1.fill_between(x, (mean-err)/plot_binwidth, (mean+err)/plot_binwidth, color=colors[i%len(colors)], alpha=0.5)
            ax1.set_xlim(time_min, time_max)
            ax2.hist(x, bins=edges, weights=counts, histtype='step',
                     color=colors[i%len(colors)], label=f'{plt_names[i]} {radius_text[i]}')
            ax2.plot(x, np.full_like(x, mean), color=colors[i%len(colors)], linestyle='dashed')
            ax2.fill_between(x, (mean-err), (mean+err), color=colors[i%len(colors)], alpha=0.5)
            ax2.set_xlim(time_min, time_max)
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel(f'Number of hits/{plot_binwidth:.2f} ns')
        ax1.legend(loc='upper right', fontsize=10)
        if y_max is not None:
            ax1.set_ylim(0, y_max/plot_binwidth)
        fig1.savefig(os.path.join(outdir, 'SiVertexBarrel_layers_hits_per_ns.png'), bbox_inches='tight')
        fig1.savefig(os.path.join(outdir, 'SiVertexBarrel_layers_hits_per_ns.pdf'), bbox_inches='tight')
        plt.close(fig1)

        ax2.set_xlabel('Time (ns)')
        ax2.set_ylabel('Number of hits')
        ax2.legend(loc='upper right', fontsize=10)
        if y_max is not None:
            ax2.set_ylim(0, y_max)
        fig2.savefig(os.path.join(outdir, 'SiVertexBarrel_layers_hits.png'), bbox_inches='tight')
        fig2.savefig(os.path.join(outdir, 'SiVertexBarrel_layers_hits.pdf'), bbox_inches='tight')
        plt.close(fig2)

        # Area-normalized figures: hits/(ns*mm^2) and hits/mm^2
        fig3, ax3 = plt.subplots(figsize=(8,5))
        fig4, ax4 = plt.subplots(figsize=(8,5))
        short_names = ['all','1st','2nd','3rd','4th','5th']
        for i, hname in enumerate(hist_names_area):
            if hname not in f:
                continue
            counts, edges = f[hname].to_numpy()
            base_bw = edges[1] - edges[0]
            ratio = int(round(plot_binwidth / base_bw))
            counts = counts.reshape(-1, ratio).sum(axis=1)
            edges = edges[::ratio]
            x = 0.5*(edges[:-1]+edges[1:])
            # area per layer i: i from 0..4 corresponds to layers 1..5
            a_mm2 = areas[i]
            counts_scaled = counts / a_mm2
            mask = (x >= time_min+5) & (x <= time_max-5)
            mean, err = _fit_const(x[mask], counts_scaled[mask])
            ax3.hist(x, bins=edges, weights=counts_scaled/plot_binwidth, histtype='step',
                     color=colors[(i+1)%len(colors)], label=f'{short_names[i+1]} {radius_text[i+1]}')
            ax3.plot(x, np.full_like(x, mean)/plot_binwidth, color=colors[(i+1)%len(colors)], linestyle='dashed')
            ax3.fill_between(x, (mean-err)/plot_binwidth, (mean+err)/plot_binwidth, color=colors[(i+1)%len(colors)], alpha=0.5)
            ax3.set_xlim(time_min, time_max)

            ax4.hist(x, bins=edges, weights=counts_scaled, histtype='step',
                     color=colors[(i+1)%len(colors)], label=f'{short_names[i+1]} {radius_text[i+1]}')
            ax4.plot(x, np.full_like(x, mean), color=colors[(i+1)%len(colors)], linestyle='dashed')
            ax4.fill_between(x, (mean-err), (mean+err), color=colors[(i+1)%len(colors)], alpha=0.5)
            ax4.set_xlim(time_min, time_max)
        ax3.set_xlabel('Time (ns)')
        ax3.set_ylabel(r'Number of hits per unit area and time [ns$^{-1}$ mm$^{-2}$]')
        ax3.legend(loc='upper right', fontsize=10)
        if y_max is not None:
            ax3.set_ylim(0, (y_max/plot_binwidth)/ (areas[1] if areas[1] else 1.0))
        fig3.savefig(os.path.join(outdir, 'SiVertexBarrel_layers_hits_per_ns_per_mm2.png'), bbox_inches='tight')
        fig3.savefig(os.path.join(outdir, 'SiVertexBarrel_layers_hits_per_ns_per_mm2.pdf'), bbox_inches='tight')
        plt.close(fig3)

        ax4.set_xlabel('Time (ns)')
        ax4.set_ylabel('Number of hits per unit area [mm$^{-2}$]')
        ax4.legend(loc='upper right', fontsize=10)
        fig4.savefig(os.path.join(outdir, 'SiVertexBarrel_layers_hits_per_mm2.png'), bbox_inches='tight')
        fig4.savefig(os.path.join(outdir, 'SiVertexBarrel_layers_hits_per_mm2.pdf'), bbox_inches='tight')
        plt.close(fig4)


def main():
    parser = argparse.ArgumentParser(description="EDM4hep timing histogrammer + plotting (Python)")
    parser.add_argument("--base-dir", required=True, help="Directory containing EDM4hep ROOT files")
    parser.add_argument(
        "--pattern",
        default="ddsim_C3_250_PS1_v2_seed_*.edm4hep.root",
        help="Glob pattern (under base-dir) to discover files (default: %(default)s)",
    )
    parser.add_argument("--bunch-spacing", type=float, default=5.25, help="Bunch spacing in ns (default: %(default)s)")
    parser.add_argument("--out", default="timing_all_BX_histograms.npz", help="Output npz file for histograms")
    parser.add_argument("--outdir", default=".", help="Directory to write PNG/PDF plots")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files (for quick tests)")
    parser.add_argument("--time-min", type=float, default=0.0, help="X-axis minimum for plotting and base histogram")
    parser.add_argument("--time-max", type=float, default=800.0, help="X-axis maximum for plotting and base histogram")
    parser.add_argument("--base-binwidth", type=float, default=0.25, help="Base bin width (ns) for accumulation")
    parser.add_argument("--plot-binwidth", type=float, default=10.0, help="Bin width (ns) for plotting and fit")
    parser.add_argument("--collider", default="C3 250 PS1", help="Collider scenario label for title")
    parser.add_argument("--detector", default="SiD_o2_v04", help="Detector version label for title")
    parser.add_argument("--load-hist", default=None, help="Load histograms from an NPZ (skip reading ROOT files)")
    parser.add_argument("--use-cache", action='store_true', help="If --out exists and matches config, reuse it")
    parser.add_argument("--vb-layers-root", default=None, help="ROOT file with SiVertexBarrel layer histos to overlay")
    parser.add_argument("--tracker-overlay", action='store_true', help="Produce Timing_All overlay for tracker-like detectors")
    parser.add_argument("--overlay-ymax", type=float, default=None, help="Y max for tracker overlay plot (hits/ns)")
    parser.add_argument("--legend-fs", type=int, default=10, help="Legend font size for all plots")
    parser.add_argument("--hpp-file", default=None, help="Path to HPP edm4hep ROOT file (many events)")
    parser.add_argument("--hpp-mu", type=float, default=None, help="Expected mean number of HPP events per bunch crossing")
    # On-the-fly SiVertexBarrel layer plots
    parser.add_argument("--vb-layers", action='store_true', help="Compute SiVertexBarrel layer timing on-the-fly from hit radii")
    parser.add_argument("--vb-ymax", type=float, default=None, help="Y max for vertex barrel layer plots (hits/ns)")
    # energy thresholds (per collection); values in same units as stored energy (usually GeV)
    parser.add_argument("--thr-vertex-barrel", type=float, default=0.0, help="Energy threshold for SiVertexBarrelHits")
    parser.add_argument("--thr-vertex-endcap", type=float, default=0.0, help="Energy threshold for SiVertexEndcapHits")
    parser.add_argument("--thr-tracker-barrel", type=float, default=0.0, help="Energy threshold for SiTrackerBarrelHits")
    parser.add_argument("--thr-tracker-endcap", type=float, default=0.0, help="Energy threshold for SiTrackerEndcapHits")
    parser.add_argument("--thr-tracker-forward", type=float, default=0.0, help="Energy threshold for SiTrackerForwardHits")
    parser.add_argument("--thr-ecal-barrel", type=float, default=0.0, help="Energy threshold for ECalBarrelHits")
    parser.add_argument("--thr-ecal-endcap", type=float, default=0.0, help="Energy threshold for ECalEndcapHits")
    parser.add_argument("--thr-hcal-barrel", type=float, default=0.0, help="Energy threshold for HCalBarrelHits")
    parser.add_argument("--thr-hcal-endcap", type=float, default=0.0, help="Energy threshold for HCalEndcapHits")
    parser.add_argument("--thr-muon-barrel", type=float, default=0.0, help="Energy threshold for MuonBarrelHits")
    parser.add_argument("--thr-muon-endcap", type=float, default=0.0, help="Energy threshold for MuonEndcapHits")
    parser.add_argument("--thr-lumical", type=float, default=0.0, help="Energy threshold for LumiCalHits")
    parser.add_argument("--thr-beamcal", type=float, default=0.0, help="Energy threshold for BeamCalHits")
    parser.add_argument("--rates-json", default=None, help="Optional path to write IPC/HPP hit rates JSON")
    args = parser.parse_args()

    # Option 1: load precomputed histograms

    if args.load_hist:
        edges, counts_total, counts_ipc, counts_hpp, raw_entry_counts, cached_files, hpp_counts_per_bx = load_histogram_npz(args.load_hist)
        files = [(idx, path) for idx, path in enumerate(cached_files)]
        print(f"Loaded histograms from {args.load_hist}; skipping ROOT reads.")
        files_to_process: list[tuple[int, str]] = []
    else:
        files = discover_files(args.base_dir, args.pattern)
        if args.max_files is not None:
            files = files[: args.max_files]
        if not files:
            raise SystemExit("No files found. Check --base-dir and --pattern.")
        edges = make_bins(args.time_min, args.time_max, args.base_binwidth)
        counts_total = {b: np.zeros(edges.size - 1, dtype=np.float64) for b in ALL_BRANCHES}
        counts_ipc = {b: np.zeros(edges.size - 1, dtype=np.float64) for b in ALL_BRANCHES}
        counts_hpp = {b: np.zeros(edges.size - 1, dtype=np.float64) for b in ALL_BRANCHES}
        raw_entry_counts = {b: 0.0 for b in ALL_BRANCHES}
        hpp_counts_per_bx: dict[str, np.ndarray] = {}
        if args.use_cache and os.path.exists(args.out):
            try:
                with np.load(args.out, allow_pickle=True) as npz:
                    edges_cached = npz["edges"]
                    files_cached = list(npz.get("files", []))
                    filenames_match = all(
                        os.path.basename(a[1]) == os.path.basename(b)
                        for a, b in zip(files, files_cached)
                    ) if files_cached else False
                    if np.allclose(edges_cached, edges) and len(files_cached) == len(files) and filenames_match:
                        print(f"Using cache from {args.out}")
                        cached_edges, cached_total, cached_ipc, cached_hpp, cached_raw, cached_files, cached_per_bx = load_histogram_npz(args.out)
                        edges = cached_edges
                        counts_total.update(cached_total)
                        counts_ipc.update(cached_ipc)
                        counts_hpp.update(cached_hpp)
                        raw_entry_counts.update(cached_raw)
                        files = [(i, p) for i, p in enumerate(cached_files)]
                        hpp_counts_per_bx = cached_per_bx
                        files_to_process = []
                    else:
                        files_to_process = files
            except Exception:
                files_to_process = files
        else:
            files_to_process = files

    vb_raw_x_list: list[np.ndarray] = []
    vb_raw_y_list: list[np.ndarray] = []
    vb_raw_t_list: list[np.ndarray] = []

    # Process ROOT files if needed
    if files_to_process:
        file_iterable = files_to_process
        for ifile, (seed, fp) in enumerate(file_iterable):
            print(f"Processing file {ifile} with seed = {seed}: {fp}")
            try:
                with uproot.open(fp) as f:
                    tree = f["events"]
                    # Trackers
                    for b in TRACKER_BRANCHES:
                        try:
                            # For SiVertexBarrelHits we may also need x,y to compute per-layer later
                            if b == "SiVertexBarrelHits" and args.vb_layers:
                                thr = {
                                    "SiVertexBarrelHits": args.thr_vertex_barrel,
                                    "SiVertexEndcapHits": args.thr_vertex_endcap,
                                    "SiTrackerBarrelHits": args.thr_tracker_barrel,
                                    "SiTrackerEndcapHits": args.thr_tracker_endcap,
                                    "SiTrackerForwardHits": args.thr_tracker_forward,
                                }[b]
                                x, y, z, t, raw_hits = read_tracker_xyz_time(
                                    tree, b, thr, return_raw_count=True
                                )
                                print(f"    SiVertexBarrelHits.time entries: {raw_hits}")
                                if t.size:
                                    t = t + ifile * args.bunch_spacing
                                    c, _ = np.histogram(t, bins=edges)
                                    counts_total[b] += c
                                    counts_ipc[b] += c
                                    # accumulate raw x, y, t for layer plots
                                    vb_raw_x_list.append(x)
                                    vb_raw_y_list.append(y)
                                    vb_raw_t_list.append(t)
                                raw_entry_counts[b] += float(raw_hits)
                            else:
                                thr = {
                                    "SiVertexBarrelHits": args.thr_vertex_barrel,
                                    "SiVertexEndcapHits": args.thr_vertex_endcap,
                                    "SiTrackerBarrelHits": args.thr_tracker_barrel,
                                    "SiTrackerEndcapHits": args.thr_tracker_endcap,
                                    "SiTrackerForwardHits": args.thr_tracker_forward,
                                }[b]
                                t, raw_hits = read_tracker_times(
                                    tree, b, thr, return_raw_count=True
                                )
                                if b == "SiVertexBarrelHits":
                                    print(f"    SiVertexBarrelHits.time entries: {raw_hits}")
                                if t.size:
                                    t = t + ifile * args.bunch_spacing
                                    c, _ = np.histogram(t, bins=edges)
                                    counts_total[b] += c
                                    counts_ipc[b] += c
                                raw_entry_counts[b] += float(raw_hits)
                        except KeyError:
                            continue
                    # Calorimeters
                    for b in CALO_BRANCHES:
                        try:
                            thr = {
                                "ECalBarrelHits": args.thr_ecal_barrel,
                                "ECalEndcapHits": args.thr_ecal_endcap,
                                "HCalBarrelHits": args.thr_hcal_barrel,
                                "HCalEndcapHits": args.thr_hcal_endcap,
                                "MuonBarrelHits": args.thr_muon_barrel,
                                "MuonEndcapHits": args.thr_muon_endcap,
                                "LumiCalHits": args.thr_lumical,
                                "BeamCalHits": args.thr_beamcal,
                            }[b]
                            t, raw_hits = read_calo_times(
                                tree, b, thr, return_raw_count=True
                            )
                            if t.size:
                                t = t + ifile * args.bunch_spacing
                                c, _ = np.histogram(t, bins=edges)
                                counts_total[b] += c
                                counts_ipc[b] += c
                            raw_entry_counts[b] += float(raw_hits)
                        except KeyError:
                            continue
            except Exception as e:
                print(f"Warning: failed to read {fp}: {e}")
                continue

    os.makedirs(args.outdir, exist_ok=True)

    # If requested, process HPP file (many events). Build per-BX histogram and then replicate across bunches.
    hpp_counts: dict[str, np.ndarray] | None = None
    num_hpp_events = 0
    if args.hpp_file is not None:
        print(f"Starting HPP background processing from: {args.hpp_file}")
        hpp_counts = {b: np.zeros(edges.size - 1, dtype=np.float64) for b in ALL_BRANCHES}
        try:
            with uproot.open(args.hpp_file) as hf:
                htree = hf["events"]
                num_hpp_events = htree.num_entries
                print(f"HPP file contains {num_hpp_events} events")
                step = min(10000, num_hpp_events)
                for start in range(0, num_hpp_events,    step):
                    stop = min(start + step, num_hpp_events)
                    # Trackers
                    for b in TRACKER_BRANCHES:
                        try:
                            thr = {
                                "SiVertexBarrelHits": args.thr_vertex_barrel,
                                "SiVertexEndcapHits": args.thr_vertex_endcap,
                                "SiTrackerBarrelHits": args.thr_tracker_barrel,
                                "SiTrackerEndcapHits": args.thr_tracker_endcap,
                                "SiTrackerForwardHits": args.thr_tracker_forward,
                            }[b]
                            names = [f"{b}", f"{b}.position.x", f"{b}.position.y", f"{b}.position.z", f"{b}.time"]
                            energy_leaf = _find_tracker_energy_leaf(htree, b)
                            if energy_leaf is not None:
                                names.append(energy_leaf)
                            arr = htree.arrays(names, entry_start=start, entry_stop=stop, library="ak")
                            posx = ak.flatten(arr[f"{b}.position.x"]) 
                            posy = ak.flatten(arr[f"{b}.position.y"]) 
                            posz = ak.flatten(arr[f"{b}.position.z"]) 
                            hit_time = ak.flatten(arr[f"{b}.time"]) 
                            energy = ak.flatten(arr[energy_leaf]) if energy_leaf is not None else None
                            # we filter out hits where ALL coordinates are zero
                            mask = ak.full_like(posx, True, dtype=np.bool_)
                            #mask = ~((posx == 0) & (posy == 0) & (posz == 0))
                            # mask = (posx != 0) & (posy != 0) & (posz != 0)
                            if energy is not None and thr > 0.0:
                                mask = mask & (energy >= thr)
                            if ak.sum(mask) > 0:
                                t = ak.to_numpy(hit_time[mask]).astype(np.float64, copy=False)
                                c, _ = np.histogram(t, bins=edges)
                                hpp_counts[b] += c
                        except KeyError:
                            continue
                    # Calorimeters
                    for b in CALO_BRANCHES:
                        try:
                            thr = {
                                "ECalBarrelHits": args.thr_ecal_barrel,
                                "ECalEndcapHits": args.thr_ecal_endcap,
                                "HCalBarrelHits": args.thr_hcal_barrel,
                                "HCalEndcapHits": args.thr_hcal_endcap,
                                "MuonBarrelHits": args.thr_muon_barrel,
                                "MuonEndcapHits": args.thr_muon_endcap,
                                "LumiCalHits": args.thr_lumical,
                                "BeamCalHits": args.thr_beamcal,
                            }[b]
                            names = [
                                f"{b}", f"{b}.position.x", f"{b}.position.y", f"{b}.position.z",
                                f"{b}.contributions_begin", f"{b}.contributions_end",
                                f"{b}Contributions.time",
                            ]
                            energy_leaf = f"{b}Contributions.energy"
                            if energy_leaf in set(htree.keys()):
                                names.append(energy_leaf)
                            else:
                                energy_leaf = None
                            arr = htree.arrays(names, entry_start=start, entry_stop=stop, library="ak")
                            posx = arr[f"{b}.position.x"]
                            posy = arr[f"{b}.position.y"]
                            posz = arr[f"{b}.position.z"]
                            beg = arr[f"{b}.contributions_begin"]
                            end = arr[f"{b}.contributions_end"]
                            ctime = arr[f"{b}Contributions.time"]
                            energy = (arr[energy_leaf] if energy_leaf is not None else None)
                            # we filter out hits where ALL coordinates are zero
                            hit_valid = ak.full_like(posx, True, dtype=np.bool_)
                            #hit_valid = ~((posx == 0) & (posy == 0) & (posz == 0)) & (beg < end)
                            # hit_valid = (posx != 0) & (posy != 0) & (posz != 0) & (beg < end)
                            hit_valid = hit_valid & (beg < end)
                            if energy is not None and thr > 0.0:
                                first_energy = energy[beg]
                                hit_valid = hit_valid & (first_energy >= thr)
                            if ak.sum(hit_valid) > 0:
                                first_time = ctime[beg]
                                t = ak.to_numpy(ak.flatten(first_time[hit_valid])).astype(np.float64, copy=False)
                                c, _ = np.histogram(t, bins=edges)
                                hpp_counts[b] += c
                        except KeyError:
                            continue
                print("Completed HPP processing")
        except Exception as e:
            print(f"Warning: failed to read HPP file {args.hpp_file}: {e}")
            hpp_counts = None

    # Normalize HPP per-BX and replicate across bunches, then sum into IPC counts
    if hpp_counts is not None:
        if args.hpp_mu is None:
            print("Warning: --hpp-file provided without --hpp-mu; HPP will not be scaled per BX.")
            hpp_scale = 1.0
        else:
            if num_hpp_events <= 0:
                print("Warning: HPP file had zero events; skipping HPP contribution.")
                hpp_counts = None
            else:
                hpp_scale = float(args.hpp_mu) / float(num_hpp_events)
                print(f"HPP normalization: mu={args.hpp_mu} events/BX, {num_hpp_events} events total, per-BX scale = {hpp_scale:.6g}")
        if hpp_counts is not None:
            # scaled per-BX counts per branch
            for b in ALL_BRANCHES:
                hpp_counts[b] = hpp_counts[b] * hpp_scale
            # replicate across bunches by bin shifts and add to counts
            base_bw = edges[1] - edges[0]
            shift_bins = int(round(args.bunch_spacing / base_bw)) if base_bw > 0 else 0
            # number of bunches displayed equals number of IPC files considered
            display_bunches = args.max_files if args.max_files is not None else len(files)
            for b in ALL_BRANCHES:
                per_bx = hpp_counts[b]
                total = counts_total[b]
                for ib in range(display_bunches):
                    shift = ib * shift_bins
                    if shift >= total.size:
                        break
                    end = min(shift + per_bx.size, total.size)
                    span = end - shift
                    if span > 0:
                        total[shift:end] += per_bx[:span]
                counts_total[b] = total
            # replicate into counts_hpp for plotting (single BX per_bx at BX0 with repeats)
            counts_hpp = accumulate_hpp_counts(hpp_counts, edges, ALL_BRANCHES, args.bunch_spacing, display_bunches)

    # Save plots and numpy arrays
    # Determine display bunch count (use --max-files if given, otherwise files processed)
    display_bunches = args.max_files if args.max_files is not None else len(files)

    plot_configs = [
        (counts_total, None, None, "Total"),
        (counts_ipc, "IPC only", "IPC", "IPC"),
        (counts_hpp, "HPP only", "HPP", "HPP"),
    ]

    rate_results: dict[str, dict] = {}
    collider_label, scenario_label = _parse_collider_label(args.collider)

    for dataset, variant_label, suffix, log_label in plot_configs:
        rate_key = (log_label or "Total").upper()
        rate_entry = rate_results.setdefault(rate_key, {})
        for b, counts in dataset.items():
            png, pdf = plot_hist(
                b,
                counts,
                edges,
                args.outdir,
                plot_xmin=args.time_min,
                plot_xmax=args.time_max,
                plot_binwidth=args.plot_binwidth,
                collider=args.collider,
                detector_version=args.detector,
                n_bunches=display_bunches,
                legend_fs=args.legend_fs,
                variant_label=variant_label,
                filename_suffix=suffix,
            )
            if suffix:
                print(f"Wrote {log_label} plot for {b} to {png} and {pdf}")
            else:
                print(f"Wrote plot for {b} to {png} and {pdf}")

            stats, status = compute_rate_stats(counts, edges, args.time_min, args.time_max, args.plot_binwidth)
            if stats is None:
                rate_entry.setdefault("_warnings", []).append({
                    "branch": b,
                    "reason": status,
                })
                continue
            location_cat = BRANCH_TO_RATE_CATEGORY.get(b)
            if location_cat is None:
                rate_entry.setdefault("_warnings", []).append({
                    "branch": b,
                    "reason": "no_category_mapping",
                })
                continue
            location, cat = location_cat
            location_block = rate_entry.setdefault(location, {})
            cat_block = location_block.setdefault(cat, {})
            collider_block = cat_block.setdefault(collider_label, {})
            collider_block[scenario_label] = {
                "value": _format_json_number(stats[0]),
                "uncertainty": _format_json_number(stats[1]),
            }

    # Save all histograms to a single NPZ for later inspection
    np.savez(
        args.out,
        edges=edges,
        files=np.array([p for _, p in files], dtype=object),
        **{f"counts_{b}": v for b, v in counts_total.items()},
        **{f"counts_ipc_{b}": counts_ipc[b] for b in ALL_BRANCHES},
        **{f"counts_hpp_{b}": counts_hpp[b] for b in ALL_BRANCHES},
        **{f"raw_entries_{b}": raw_entry_counts[b] for b in ALL_BRANCHES},
        **({} if hpp_counts is None else {f"hpp_counts_per_bx_{b}": v for b, v in hpp_counts.items()}),
        hpp_file=(args.hpp_file if args.hpp_file is not None else ''),
        hpp_mu=(args.hpp_mu if args.hpp_mu is not None else np.nan),
        collider_label=args.collider,
        detector_label=args.detector,
    )
    print(f"Wrote histograms to {args.out} and plots to {args.outdir}")

    if args.rates_json:
        output_payload = {
            "IPC": rate_results.get("IPC", {}),
            "HPP": rate_results.get("HPP", {}),
            "TOTAL": rate_results.get("TOTAL", {}),
        }
        if os.path.exists(args.rates_json):
            try:
                with open(args.rates_json, "r", encoding="utf-8") as f:
                    existing_payload = json.load(f)
                _merge_nested_dict(existing_payload, output_payload)
                output_payload = existing_payload
            except Exception as exc:
                print(f"Warning: failed to read existing rates JSON ({args.rates_json}): {exc}")
        try:
            with open(args.rates_json, "w", encoding="utf-8") as f:
                json.dump(output_payload, f, indent=2, sort_keys=True)
            print(f"Wrote rates JSON to {args.rates_json}")
        except Exception as exc:
            print(f"Warning: failed to write rates JSON ({args.rates_json}): {exc}")

    # Compare integrated hits to the constant-fit expectation for each branch.
    time_span = edges[-1] - edges[0] if edges.size > 1 else 0.0
    if time_span <= 0:
        print("Warning: non-positive time span; skipping hit comparison summary.")
    else:
        print(f"Hit comparison over {time_span:.2f} ns (time_max = {args.time_max:.2f} ns):")
        for b, counts in counts_total.items():
            total_hits = float(np.sum(counts))
            entries = float(raw_entry_counts.get(b, float('nan')))
            nbx = display_bunches if display_bunches > 0 else 1
            avg_entries = entries / nbx if np.isfinite(entries) else float('nan')
            try:
                counts_reb, edges_reb = rebin_counts(counts, edges, args.plot_binwidth, args.time_min, args.time_max)
            except ValueError:
                print(f"  {b}: summed = {total_hits:.2f} hits, entries = {entries:.0f}, entries/BX = {avg_entries:.2f}; unable to rebin with plot_binwidth = {args.plot_binwidth}; skipping fit estimate.")
                continue
            centers = 0.5 * (edges_reb[:-1] + edges_reb[1:])
            fit_mask = (centers >= (edges[0] + 5.0)) & (centers <= (edges[-1] - 5.0))
            if not bool(fit_mask.size and fit_mask.any()):
                print(f"  {b}: summed = {total_hits:.2f} hits, entries = {entries:.0f}, entries/BX = {avg_entries:.2f}; insufficient bins for constant fit")
                continue
            mean_hits_per_bin, _ = _fit_const(centers[fit_mask], counts_reb[fit_mask])
            mean_hits_per_ns = mean_hits_per_bin / args.plot_binwidth if args.plot_binwidth > 0 else float('nan')
            fit_hits = mean_hits_per_ns * time_span if np.isfinite(mean_hits_per_ns) else float('nan')
            if np.isfinite(fit_hits):
                print(
                    f"  {b}: summed = {total_hits:.2f} hits, fit estimate = {fit_hits:.2f} hits (mean = {mean_hits_per_ns:.4f} hits/ns), entries = {entries:.0f}, entries/BX = {avg_entries:.2f}"
                )
            else:
                print(f"  {b}: summed = {total_hits:.2f} hits, entries = {entries:.0f}, entries/BX = {avg_entries:.2f}; fit estimate unavailable")

    # Optional: tracker overlay plot
    if args.tracker_overlay:
        plot_tracker_overlay({b: counts_total[b] for b in counts_total}, edges, args.outdir,
                             time_min=args.time_min, time_max=args.time_max,
                             plot_binwidth=args.plot_binwidth, y_max=args.overlay_ymax,
                             legend_fs=args.legend_fs)
        print(f"Wrote overlay to {args.outdir}/Timing_All.png and Timing_All.pdf")

    # Optional: SiVertexBarrel layer plots from an existing ROOT file
    if args.vb_layers_root and os.path.exists(args.vb_layers_root):
        plot_vertex_barrel_layers_from_root(
            args.vb_layers_root, args.outdir,
            time_min=args.time_min, time_max=args.time_max,
            plot_binwidth=args.plot_binwidth,
            y_max=args.overlay_ymax,
        )
        print("Wrote SiVertexBarrel layer comparative plots.")

    # Optional: on-the-fly SiVertexBarrel layer plots using hit radii
    if args.vb_layers and 'vb_raw_t_list' in locals() and len(vb_raw_t_list) > 0:
        try:
            import mplhep as hep_style
            hep_style.style.use("CMS")
        except Exception:
            pass
        x_all = np.concatenate(vb_raw_x_list) if len(vb_raw_x_list) else np.empty(0)
        y_all = np.concatenate(vb_raw_y_list) if len(vb_raw_y_list) else np.empty(0)
        t_all = np.concatenate(vb_raw_t_list) if len(vb_raw_t_list) else np.empty(0)
        r_all = np.sqrt(x_all * x_all + y_all * y_all)

        # Build per-layer masks
        masks = []
        for i in range(5):
            if i == 0:
                m = (r_all < VB_UPPER_R[i]) & (r_all >= VB_LOWER_R[i])
            else:
                m = (r_all >= VB_LOWER_R[i]) & (r_all <= VB_UPPER_R[i])
            masks.append(m)

        # Figure 1: hits/ns per layer (including all-layers overlay)
        fig1, ax1 = plt.subplots(figsize=(8,5))
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        labels = ['all', '1st', '2nd', '3rd', '4th', '5th']
        radii_text = [' ', 'r=15.05 mm', 'r=23.03 mm', 'r=35.79 mm', 'r=47.50 mm', 'r=59.90 mm']

        # All layers first
        c_all, e_all = np.histogram(t_all, bins=edges)
        c_reb, e_reb = rebin_counts(c_all, edges, args.plot_binwidth, args.time_min, args.time_max)
        xcent = 0.5*(e_reb[:-1] + e_reb[1:])
        mask_fit = (xcent >= args.time_min + 5) & (xcent <= args.time_max - 5)
        mean_all, err_all = _fit_const(xcent[mask_fit], c_reb[mask_fit])
        ax1.hist(xcent, bins=e_reb, weights=c_reb/args.plot_binwidth, histtype='step', color=colors[0], label='all layers')
        ax1.plot(xcent, np.full_like(xcent, mean_all)/args.plot_binwidth, color=colors[0], linestyle='dashed')
        ax1.fill_between(xcent, (mean_all-err_all)/args.plot_binwidth, (mean_all+err_all)/args.plot_binwidth, color=colors[0], alpha=0.4)

        # Layers 1..5
        areas_mm2 = 2 * np.pi * VB_RADIAL_CENTERS * VB_ACTIVE_LENGTH_MM
        for i, m in enumerate(masks, start=1):
            t_layer = t_all[m]
            c, _ = np.histogram(t_layer, bins=edges)
            c_reb, e_reb = rebin_counts(c, edges, args.plot_binwidth, args.time_min, args.time_max)
            xcent = 0.5*(e_reb[:-1] + e_reb[1:])
            mask_fit = (xcent >= args.time_min + 5) & (xcent <= args.time_max - 5)
            mean, err = _fit_const(xcent[mask_fit], c_reb[mask_fit])
            ax1.hist(xcent, bins=e_reb, weights=c_reb/args.plot_binwidth, histtype='step', color=colors[i%len(colors)], label=f'{labels[i]} layer {radii_text[i]}')
            ax1.plot(xcent, np.full_like(xcent, mean)/args.plot_binwidth, color=colors[i%len(colors)], linestyle='dashed')
            ax1.fill_between(xcent, (mean-err)/args.plot_binwidth, (mean+err)/args.plot_binwidth, color=colors[i%len(colors)], alpha=0.4)

        ax1.set_xlim(args.time_min, args.time_max)
        ax1.set_xlabel('Time (ns)', fontsize=12)
        ax1.set_ylabel(f'Number of hits per ns (bin = {args.plot_binwidth:.2f} ns)', fontsize=12)
        # titles to match other plots
        bunch_text = f"{display_bunches} bunches" if display_bunches != 1 else "1 bunch"
        left_title = f"{args.collider} ({bunch_text})"
        right_title = f"{args.detector} - SiVertexBarrelHits layers"
        ax1.set_title(left_title, fontsize=15, loc='left')
        ax1.set_title(right_title, fontsize=15, loc='right')
        ax1.tick_params(labelsize=11)
        if args.vb_ymax is not None:
            ax1.set_ylim(0, args.vb_ymax)
        ax1.legend(loc='upper right', fontsize=args.legend_fs)
        fig1.savefig(os.path.join(args.outdir, 'SiVertexBarrel_layers_hits_per_ns.png'), bbox_inches='tight')
        fig1.savefig(os.path.join(args.outdir, 'SiVertexBarrel_layers_hits_per_ns.pdf'), bbox_inches='tight')
        plt.close(fig1)

        # Figure 2: hits (no per-ns)
        fig2, ax2 = plt.subplots(figsize=(8,5))
        ax2.hist(xcent, bins=e_reb, weights=c_reb, histtype='step', color=colors[0], label='all layers')
        ax2.plot(xcent, np.full_like(xcent, mean_all), color=colors[0], linestyle='dashed')
        ax2.fill_between(xcent, (mean_all-err_all), (mean_all+err_all), color=colors[0], alpha=0.4)
        for i, m in enumerate(masks, start=1):
            t_layer = t_all[m]
            c, _ = np.histogram(t_layer, bins=edges)
            c_reb, e_reb = rebin_counts(c, edges, args.plot_binwidth, args.time_min, args.time_max)
            xcent = 0.5*(e_reb[:-1] + e_reb[1:])
            mask_fit = (xcent >= args.time_min + 5) & (xcent <= args.time_max - 5)
            mean, err = _fit_const(xcent[mask_fit], c_reb[mask_fit])
            ax2.hist(xcent, bins=e_reb, weights=c_reb, histtype='step', color=colors[i%len(colors)], label=f'{labels[i]} layer {radii_text[i]}')
            ax2.plot(xcent, np.full_like(xcent, mean), color=colors[i%len(colors)], linestyle='dashed')
            ax2.fill_between(xcent, (mean-err), (mean+err), color=colors[i%len(colors)], alpha=0.4)
        ax2.set_xlim(args.time_min, args.time_max)
        ax2.set_xlabel('Time (ns)', fontsize=12)
        ax2.set_ylabel(f'Number of hits per {args.plot_binwidth:.2f} ns bin', fontsize=12)
        # titles to match other plots
        bunch_text = f"{display_bunches} bunches" if display_bunches != 1 else "1 bunch"
        left_title = f"{args.collider} ({bunch_text})"
        right_title = f"{args.detector} - SiVertexBarrelHits layers"
        ax2.set_title(left_title, fontsize=15, loc='left')
        ax2.set_title(right_title, fontsize=15, loc='right')
        ax2.tick_params(labelsize=11)
        if args.vb_ymax is not None:
            ax2.set_ylim(0, args.vb_ymax * args.plot_binwidth)
        ax2.legend(loc='upper right', fontsize=args.legend_fs)
        fig2.savefig(os.path.join(args.outdir, 'SiVertexBarrel_layers_hits.png'), bbox_inches='tight')
        fig2.savefig(os.path.join(args.outdir, 'SiVertexBarrel_layers_hits.pdf'), bbox_inches='tight')
        plt.close(fig2)

        # Figure 3: hits/(ns·mm^2)
        fig3, ax3 = plt.subplots(figsize=(8,5))
        for i, m in enumerate(masks, start=1):
            t_layer = t_all[m]
            c, _ = np.histogram(t_layer, bins=edges)
            c_reb, e_reb = rebin_counts(c, edges, args.plot_binwidth, args.time_min, args.time_max)
            xcent = 0.5*(e_reb[:-1] + e_reb[1:])
            a_mm2 = 2 * np.pi * VB_RADIAL_CENTERS[i-1] * VB_ACTIVE_LENGTH_MM
            y = (c_reb / args.plot_binwidth) / a_mm2
            mask_fit = (xcent >= args.time_min + 5) & (xcent <= args.time_max - 5)
            mean, err = _fit_const(xcent[mask_fit], y[mask_fit])
            ax3.hist(xcent, bins=e_reb, weights=y, histtype='step', color=colors[i%len(colors)], label=f'{labels[i]} layer {radii_text[i]}')
            ax3.plot(xcent, np.full_like(xcent, mean), color=colors[i%len(colors)], linestyle='dashed')
            ax3.fill_between(xcent, (mean-err), (mean+err), color=colors[i%len(colors)], alpha=0.4)
        ax3.set_xlim(args.time_min, args.time_max)
        ax3.set_xlabel('Time (ns)', fontsize=12)
        ax3.set_ylabel(r'Number of hits per (ns·mm$^2$)', fontsize=12)
        # titles to match other plots
        bunch_text = f"{display_bunches} bunches" if display_bunches != 1 else "1 bunch"
        left_title = f"{args.collider} ({bunch_text})"
        right_title = f"{args.detector} - SiVertexBarrelHits layers"
        ax3.set_title(left_title, fontsize=15, loc='left')
        ax3.set_title(right_title, fontsize=15, loc='right')
        ax3.tick_params(labelsize=11)
        ax3.legend(loc='upper right', fontsize=args.legend_fs)
        fig3.savefig(os.path.join(args.outdir, 'SiVertexBarrel_layers_hits_per_ns_per_mm2.png'), bbox_inches='tight')
        fig3.savefig(os.path.join(args.outdir, 'SiVertexBarrel_layers_hits_per_ns_per_mm2.pdf'), bbox_inches='tight')
        plt.close(fig3)

        # Figure 4: hits/mm^2
        fig4, ax4 = plt.subplots(figsize=(8,5))
        for i, m in enumerate(masks, start=1):
            t_layer = t_all[m]
            c, _ = np.histogram(t_layer, bins=edges)
            a_mm2 = 2 * np.pi * VB_RADIAL_CENTERS[i-1] * VB_ACTIVE_LENGTH_MM
            y = c / a_mm2
            ax4.hist(edges[:-1], bins=edges, weights=y, histtype='step', color=colors[i%len(colors)], label=f'{labels[i]} layer {radii_text[i]}')
        ax4.set_xlim(args.time_min, args.time_max)
        ax4.set_xlabel('Time (ns)', fontsize=12)
        ax4.set_ylabel('Number of hits per mm$^2$ per bin', fontsize=12)
        # titles to match other plots
        bunch_text = f"{display_bunches} bunches" if display_bunches != 1 else "1 bunch"
        left_title = f"{args.collider} ({bunch_text})"
        right_title = f"{args.detector} - SiVertexBarrelHits layers"
        ax4.set_title(left_title, fontsize=15, loc='left')
        ax4.set_title(right_title, fontsize=15, loc='right')
        ax4.tick_params(labelsize=11)
        ax4.legend(loc='upper right', fontsize=args.legend_fs)
        fig4.savefig(os.path.join(args.outdir, 'SiVertexBarrel_layers_hits_per_mm2.png'), bbox_inches='tight')
        fig4.savefig(os.path.join(args.outdir, 'SiVertexBarrel_layers_hits_per_mm2.pdf'), bbox_inches='tight')
        plt.close(fig4)
    elif args.vb_layers:
        print("Info: --vb-layers enabled but no SiVertexBarrel hits passed selection; skipping layer plots.")


if __name__ == "__main__":
    main()

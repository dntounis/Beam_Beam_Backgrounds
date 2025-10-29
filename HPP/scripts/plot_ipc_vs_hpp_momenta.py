#!/usr/bin/env python3
import argparse
import os
import glob
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

import colorsys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

try:
    import mplhep as hep
    hep.style.use(hep.style.ATLAS)
except Exception:
    pass

# Optional heavy deps imported lazily where needed


ColliderKey = str  # e.g. "C3_250_PS1"

IPC_BASE_COLOR = "#0072B2"   # Okabe-Ito blue
HPP_BASE_COLOR = "#D55E00"   # Okabe-Ito vermillion
DEFAULT_ACCENT = "#595959"

LABEL_LINESTYLES = {
    "C3_250_PS1": "-",
    "C3_250_PS2": "-",
    "C3_550_PS1": "--",
    "C3_550_PS2": "--",
}


def adjust_lightness(color_hex: str, delta: float) -> str:
    """Shift perceived lightness without altering hue (aids color-blind palettes)."""
    r, g, b = mcolors.to_rgb(color_hex)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l + delta))
    return mcolors.to_hex(colorsys.hls_to_rgb(h, l, s))


def color_variant(base_hex: str, idx: int, total: int) -> str:
    if total <= 1:
        return base_hex
    offsets = np.linspace(-0.18, 0.28, total)
    return adjust_lightness(base_hex, float(offsets[idx % total]))


def color_for_source(source: str, idx: Optional[int] = None, total: Optional[int] = None) -> str:
    base_map = {
        "IPC": IPC_BASE_COLOR,
        "HPP": HPP_BASE_COLOR,
    }
    base = base_map.get(source, DEFAULT_ACCENT)
    if idx is None or total is None:
        return base
    return color_variant(base, idx, max(total, 1))


def linestyle_for_label(label: str) -> str:
    return LABEL_LINESTYLES.get(label, "-")


def _ensure_outdir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _read_ipc_pass_minmax(dir_path: str, pattern: str, max_files: Optional[int] = None) -> Tuple[float, float, float, float, float, float]:
    e_min, e_max = np.inf, -np.inf
    pt_min, pt_max = np.inf, -np.inf
    pz_min, pz_max = np.inf, -np.inf
    files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    if max_files is not None and max_files > 0 and len(files) > max_files:
        logging.info("IPC(minmax): limiting to %d/%d files in %s", max_files, len(files), dir_path)
        files = files[:max_files]
    for fp in files:
        try:
            with open(fp, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    try:
                        energy = abs(float(parts[0]))  # GeV
                        vx, vy, vz = map(float, parts[1:4])
                    except Exception:
                        continue
                    px = energy * vx
                    py = energy * vy
                    pz = energy * vz
                    pt = float(np.hypot(px, py))
                    # Update ranges
                    if energy < e_min: e_min = energy
                    if energy > e_max: e_max = energy
                    if pt < pt_min: pt_min = pt
                    if pt > pt_max: pt_max = pt
                    if pz < pz_min: pz_min = pz
                    if pz > pz_max: pz_max = pz
        except Exception:
            continue
    # Guard against empty
    if not np.isfinite(e_min):
        e_min, e_max = 0.0, 1.0
    if not np.isfinite(pt_min):
        pt_min, pt_max = 0.0, 1.0
    if not np.isfinite(pz_min):
        pz_min, pz_max = -1.0, 1.0
    return e_min, e_max, pt_min, pt_max, pz_min, pz_max


def _iterate_ipc_values(dir_path: str, pattern: str) -> Iterable[Tuple[float, float, float]]:
    files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    for fp in files:
        try:
            with open(fp, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    try:
                        energy = abs(float(parts[0]))
                        vx, vy, vz = map(float, parts[1:4])
                    except Exception:
                        continue
                    px = energy * vx
                    py = energy * vy
                    pz = energy * vz
                    pt = float(np.hypot(px, py))
                    yield energy, pt, pz
        except Exception:
            continue


class StreamingHist:
    def __init__(self, bins: int, range_: Tuple[float, float]):
        self.bins = bins
        self.range = range_
        self.counts = np.zeros(bins, dtype=float)
        self.edges = np.linspace(range_[0], range_[1], bins + 1)
        self.total = 0

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        c, _ = np.histogram(values, bins=self.edges)
        self.counts += c
        self.total += int(values.size)

    def finalize(self, density: bool) -> Tuple[np.ndarray, np.ndarray]:
        if density and self.total > 0:
            widths = np.diff(self.edges)
            area = float(self.counts.sum() * (widths.mean() if np.allclose(widths, widths[0]) else 1.0))
            # Proper density: divide by total counts and bin width
            densities = self.counts / (self.counts.sum() * widths)
            return 0.5 * (self.edges[:-1] + self.edges[1:]), densities
        else:
            centers = 0.5 * (self.edges[:-1] + self.edges[1:])
            return centers, self.counts.copy()


def _pad_range(lo: float, hi: float, frac: float = 0.02) -> Tuple[float, float]:
    if hi <= lo:
        if hi == lo:
            # Expand a degenerate range slightly
            delta = 1.0 if hi == 0.0 else abs(hi) * 0.1
            return lo - delta, hi + delta
        return hi, lo
    span = hi - lo
    return lo - frac * span, hi + frac * span


def _hpp_pass_minmax(root_path: str, select_final: bool) -> Tuple[float, float, float, float, float, float]:
    import uproot
    import awkward as ak

    e_min, e_max = np.inf, -np.inf
    pt_min, pt_max = np.inf, -np.inf
    pz_min, pz_max = np.inf, -np.inf

    branches = [
        "MCParticle.momentum.x",
        "MCParticle.momentum.y",
        "MCParticle.momentum.z",
        "MCParticle.mass",
        "MCParticle.PDG",
    ]
    if select_final:
        branches.append("MCParticle.generatorStatus")

    try:
        with uproot.open(root_path) as f:
            if "events" not in f:
                raise RuntimeError("'events' tree not found in file")
            tree = f["events"]
            for arrays in tree.iterate(filter_name=branches, step_size="100 MB"):
                px = arrays["MCParticle.momentum.x"]
                py = arrays["MCParticle.momentum.y"]
                pz = arrays["MCParticle.momentum.z"]
                m = arrays["MCParticle.mass"]
                if select_final:
                    status = arrays["MCParticle.generatorStatus"]
                    mask = (status == 1)
                    px = px[mask]
                    py = py[mask]
                    pz = pz[mask]
                    m = m[mask]
                pt = np.sqrt(px * px + py * py)
                p = np.sqrt(px * px + py * py + pz * pz)
                energy = np.sqrt(p * p + m * m)
                # Flatten and to numpy
                e_np = ak.to_numpy(ak.flatten(energy, axis=None))
                pt_np = ak.to_numpy(ak.flatten(pt, axis=None))
                pz_np = ak.to_numpy(ak.flatten(pz, axis=None))
                if e_np.size:
                    e_min = min(e_min, float(np.nanmin(e_np)))
                    e_max = max(e_max, float(np.nanmax(e_np)))
                if pt_np.size:
                    pt_min = min(pt_min, float(np.nanmin(pt_np)))
                    pt_max = max(pt_max, float(np.nanmax(pt_np)))
                if pz_np.size:
                    pz_min = min(pz_min, float(np.nanmin(pz_np)))
                    pz_max = max(pz_max, float(np.nanmax(pz_np)))
    except Exception as exc:
        logging.error("HPP: failed to scan ranges for %s: %s", root_path, exc)
        raise

    if not np.isfinite(e_min):
        e_min, e_max = 0.0, 1.0
    if not np.isfinite(pt_min):
        pt_min, pt_max = 0.0, 1.0
    if not np.isfinite(pz_min):
        pz_min, pz_max = -1.0, 1.0
    return e_min, e_max, pt_min, pt_max, pz_min, pz_max


def _fill_hpp_hists(root_path: str, bins: int, ranges: Dict[str, Tuple[float, float]], select_final: bool,
                    density: bool) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    import uproot
    import awkward as ak

    hE = StreamingHist(bins, ranges["energy"]) 
    hPt = StreamingHist(bins, ranges["pt"]) 
    hPz = StreamingHist(bins, ranges["pz"]) 
    pdg_total: Dict[int, float] = {}
    total_particles = 0.0
    total_events = 0

    branches = [
        "MCParticle.momentum.x",
        "MCParticle.momentum.y",
        "MCParticle.momentum.z",
        "MCParticle.mass",
        "MCParticle.PDG",
    ]
    if select_final:
        branches.append("MCParticle.generatorStatus")

    chunk_idx = 0
    try:
        with uproot.open(root_path) as f:
            if "events" not in f:
                raise RuntimeError("'events' tree not found in file")
            tree = f["events"]
            for arrays in tree.iterate(filter_name=branches, step_size="200 MB"):
                chunk_idx += 1
                if chunk_idx % 10 == 1:
                    logging.info("HPP: processed %d chunks from %s", chunk_idx, os.path.basename(root_path))
                px = arrays["MCParticle.momentum.x"]
                py = arrays["MCParticle.momentum.y"]
                pz = arrays["MCParticle.momentum.z"]
                m = arrays["MCParticle.mass"]
                if select_final:
                    status = arrays["MCParticle.generatorStatus"]
                    mask = (status == 1)
                    px = px[mask]
                    py = py[mask]
                    pz = pz[mask]
                    m = m[mask]
                pt = np.sqrt(px * px + py * py)
                p = np.sqrt(px * px + py * py + pz * pz)
                energy = np.sqrt(p * p + m * m)
                counts_per_event = ak.to_numpy(ak.num(energy, axis=1))
                if counts_per_event.size:
                    total_particles += float(counts_per_event.sum())
                    total_events += int(counts_per_event.size)
                e_np = ak.to_numpy(ak.flatten(energy, axis=None))
                pt_np = ak.to_numpy(ak.flatten(pt, axis=None))
                pz_np = ak.to_numpy(ak.flatten(pz, axis=None))
                # PDG counting (robust to uproot/awkward return types)
                try:
                    pdg_arr = arrays["MCParticle.PDG"]
                except Exception:
                    pdg_arr = None
                if pdg_arr is not None:
                    if select_final:
                        pdg_arr = pdg_arr[mask]
                    pdg_np = ak.to_numpy(ak.flatten(pdg_arr, axis=None))
                    if pdg_np.size:
                        vals, cnts = np.unique(pdg_np, return_counts=True)
                        for k, v in zip(vals, cnts):
                            pdg_total[int(k)] = pdg_total.get(int(k), 0.0) + float(v)
                if e_np.size:
                    hE.update(e_np)
                if pt_np.size:
                    hPt.update(pt_np)
                if pz_np.size:
                    hPz.update(pz_np)
    except Exception as exc:
        logging.error("HPP: failed reading %s: %s", root_path, exc)
        return {
            "energy": (np.array([]), np.array([])),
            "pt": (np.array([]), np.array([])),
            "pz": (np.array([]), np.array([])),
            "pdg_counts": {},
            "total_particles": 0.0,
            "total_events": 0,
        }

    return {
        "energy": hE.finalize(density),
        "pt": hPt.finalize(density),
        "pz": hPz.finalize(density),
        "pdg_counts": pdg_total,
        "total_particles": float(total_particles),
        "total_events": int(total_events),
    }


def _hist_finalize(edges: np.ndarray, counts: np.ndarray, density: bool) -> Tuple[np.ndarray, np.ndarray]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    if not density:
        return centers, counts.copy()
    widths = np.diff(edges)
    total = counts.sum()
    if total <= 0:
        return centers, counts.copy()
    dens = counts / (total * widths)
    return centers, dens


def _read_ipc_file_counts(fp: str, edgesE: np.ndarray, edgesPt: np.ndarray, edgesPz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, float], int]:
    e_vals: List[float] = []
    pt_vals: List[float] = []
    pz_vals: List[float] = []
    pdg_counts: Dict[int, float] = {}
    try:
        with open(fp, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 7:
                    continue
                try:
                    signed_e = float(parts[0])
                    energy = abs(signed_e)
                    vx, vy, vz = map(float, parts[1:4])
                except Exception:
                    continue
                px = energy * vx
                py = energy * vy
                pz = energy * vz
                pt = float(np.hypot(px, py))
                e_vals.append(energy)
                pt_vals.append(pt)
                pz_vals.append(pz)
                # PDG by convention: positive energy -> e- (11), negative -> e+ (-11)
                if signed_e > 0:
                    pdg_counts[11] = pdg_counts.get(11, 0.0) + 1.0
                elif signed_e < 0:
                    pdg_counts[-11] = pdg_counts.get(-11, 0.0) + 1.0
    except Exception:
        return (np.zeros(edgesE.size - 1, dtype=float),
                np.zeros(edgesPt.size - 1, dtype=float),
                np.zeros(edgesPz.size - 1, dtype=float),
                {},
                0)
    e_vals_np = np.asarray(e_vals, dtype=float)
    pt_vals_np = np.asarray(pt_vals, dtype=float)
    pz_vals_np = np.asarray(pz_vals, dtype=float)
    cE, _ = np.histogram(e_vals_np, bins=edgesE)
    cPt, _ = np.histogram(pt_vals_np, bins=edgesPt)
    cPz, _ = np.histogram(pz_vals_np, bins=edgesPz)
    total_particles = int(e_vals_np.size)
    return cE.astype(float), cPt.astype(float), cPz.astype(float), pdg_counts, total_particles


def _fill_ipc_hists(dir_path: str, pattern: str, bins: int, ranges: Dict[str, Tuple[float, float]], density: bool,
                    max_workers: int = 8, max_files: Optional[int] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    total_found = len(files)
    if max_files is not None and max_files > 0 and total_found > max_files:
        logging.info("IPC: limiting to %d/%d files in %s", max_files, total_found, dir_path)
        files = files[:max_files]
    if not files:
        logging.warning("IPC: no files matched pattern '%s' in %s", pattern, dir_path)
        return {
            "energy": (np.array([]), np.array([])),
            "pt": (np.array([]), np.array([])),
            "pz": (np.array([]), np.array([])),
        }
    logging.info("IPC: using %d files (found %d) in %s", len(files), total_found, dir_path)

    edgesE = np.linspace(ranges["energy"][0], ranges["energy"][1], bins + 1)
    edgesPt = np.linspace(ranges["pt"][0], ranges["pt"][1], bins + 1)
    edgesPz = np.linspace(ranges["pz"][0], ranges["pz"][1], bins + 1)

    countsE = np.zeros(bins, dtype=float)
    countsPt = np.zeros(bins, dtype=float)
    countsPz = np.zeros(bins, dtype=float)

    processed = 0
    total_particles = 0.0
    pdg_total: Dict[int, float] = {}
    log_step = max(1, len(files) // 20)  # ~5% steps
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_read_ipc_file_counts, fp, edgesE, edgesPt, edgesPz): fp for fp in files}
        for fut in as_completed(futures):
            cE, cPt, cPz, pdg_counts, particle_count = fut.result()
            countsE += cE
            total_particles += float(particle_count)
            countsPt += cPt
            countsPz += cPz
            for k, v in pdg_counts.items():
                pdg_total[k] = pdg_total.get(k, 0.0) + float(v)
            processed += 1
            if processed % log_step == 0 or processed == len(files):
                logging.info("IPC: processed %d/%d files (%.0f%%) in %s", processed, len(files), 100.0 * processed / len(files), dir_path)

    xE, yE = _hist_finalize(edgesE, countsE, density)
    xPt, yPt = _hist_finalize(edgesPt, countsPt, density)
    xPz, yPz = _hist_finalize(edgesPz, countsPz, density)
    return {
        "energy": (xE, yE),
        "pt": (xPt, yPt),
        "pz": (xPz, yPz),
        "pdg_counts": pdg_total,
        "total_particles": float(total_particles),
        "bunch_crossings": int(len(files)),
    }


def _make_plot(xy_ipc: Tuple[np.ndarray, np.ndarray], xy_hpp: Tuple[np.ndarray, np.ndarray], xlabel: str,
               title: str, outpath: str, logy: bool, density: bool, ymin: Optional[float] = None,
               xmin: Optional[float] = None, xmax: Optional[float] = None, linestyle: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    if xy_ipc[0].size:
        ax.plot(xy_ipc[0], xy_ipc[1], color=color_for_source("IPC"), lw=1.4, label="IPC", linestyle=(linestyle or "-"))
    if xy_hpp[0].size:
        ax.plot(xy_hpp[0], xy_hpp[1], color=color_for_source("HPP"), lw=1.4, label="HPP", linestyle=(linestyle or "-"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized counts" if density else "Counts")
    ax.grid(alpha=0.25)
    if logy:
        ax.set_yscale("log")
    if ymin is not None and np.isfinite(ymin):
        lo, hi = ax.get_ylim()
        ax.set_ylim(bottom=float(ymin), top=hi)
    cur_left, cur_right = ax.get_xlim()
    new_left = cur_left
    new_right = cur_right
    if xmin is not None and np.isfinite(xmin):
        new_left = float(xmin)
    if xmax is not None and np.isfinite(xmax):
        new_right = float(xmax)
    if np.isfinite(new_left) and np.isfinite(new_right) and new_right > new_left:
        ax.set_xlim(new_left, new_right)
    ax.legend(frameon=True)
    fig.tight_layout()
    #fig.savefig(outpath, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outpath)

    plt.close(fig)


def _pdg_to_xy(pdg_counts: Dict[int, float], density: bool) -> Tuple[np.ndarray, np.ndarray]:
    if not pdg_counts:
        return np.array([]), np.array([])
    # Sort PDGs numerically for stable plotting
    ids = np.array(sorted(pdg_counts.keys()), dtype=int)
    vals = np.array([float(pdg_counts[i]) for i in ids], dtype=float)
    if density and vals.sum() > 0:
        vals = vals / vals.sum()
    return ids, vals


def _make_pdg_plot(ipc_pdg: Dict[int, float], hpp_pdg: Dict[int, float], title: str, outpath: str, logy: bool, density: bool) -> None:
    x_ipc, y_ipc = _pdg_to_xy(ipc_pdg, density)
    x_hpp, y_hpp = _pdg_to_xy(hpp_pdg, density)
    if x_ipc.size == 0 and x_hpp.size == 0:
        return
    # Build a union of PDG ids to set x ticks
    all_ids = sorted(set(map(int, x_ipc.tolist())) | set(map(int, x_hpp.tolist())))
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    if x_ipc.size:
        ax.plot(x_ipc, y_ipc, color=color_for_source("IPC"), lw=1.4, marker="o", ms=3, label="IPC")
    if x_hpp.size:
        ax.plot(x_hpp, y_hpp, color=color_for_source("HPP"), lw=1.4, marker="o", ms=3, label="HPP")
    ax.set_xlabel("PDG ID")
    ax.set_ylabel("Fraction" if density else "Counts")
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.set_xticks(all_ids)
    ax.set_xticklabels([str(i) for i in all_ids], rotation=45, ha="right")
    ax.legend(frameon=True)
    fig.tight_layout()
    #fig.savefig(outpath, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outpath)
    plt.close(fig)


def _make_pdg_plot_with_limits(ipc_pdg: Dict[int, float], hpp_pdg: Dict[int, float], title: str, outpath: str,
                               logy: bool, density: bool, xmin: Optional[float], xmax: Optional[float],
                               linestyle: Optional[str] = None) -> None:
    x_ipc, y_ipc = _pdg_to_xy(ipc_pdg, density)
    x_hpp, y_hpp = _pdg_to_xy(hpp_pdg, density)
    if x_ipc.size == 0 and x_hpp.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    if x_ipc.size:
        ax.plot(x_ipc, y_ipc, color=color_for_source("IPC"), lw=1.4, marker="o", ms=3, label="IPC", linestyle=(linestyle or "-"))
    if x_hpp.size:
        ax.plot(x_hpp, y_hpp, color=color_for_source("HPP"), lw=1.4, marker="o", ms=3, label="HPP", linestyle=(linestyle or "-"))
    ax.set_xlabel("PDG ID")
    ax.set_ylabel("Fraction" if density else "Counts")
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    # X-limits
    left, right = ax.get_xlim()
    if xmin is not None and np.isfinite(xmin):
        left = float(xmin)
    if xmax is not None and np.isfinite(xmax):
        right = float(xmax)
    if right > left:
        ax.set_xlim(left, right)
    # Ticks: show integer ticks within range
    try:
        xticks = np.arange(int(np.ceil(left)), int(np.floor(right)) + 1)
        if xticks.size <= 50:  # avoid too many labels
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(i) for i in xticks], rotation=45, ha="right")
    except Exception:
        pass
    ax.legend(frameon=True)
    fig.tight_layout()
    #fig.savefig(outpath, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outpath)
    plt.close(fig)


def _categorize_pdg(pdg: int) -> str:
    a = abs(int(pdg))
    if a == 22:
        return "photon"
    if a in (12, 14, 16):
        return "neutrino"
    if a in (11, 13, 15):
        return "lepton"
    if a in (111, 211):
        return "pion"
    if a in (130, 310, 311, 321):
        return "kaon"
    if a in (2212, 2112):
        return "nucleon"
    # All remaining hadrons/particles → other
    return "other"


def _pdg_counts_to_categories(pdg_counts: Dict[int, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (pdg_counts or {}).items():
        cat = _categorize_pdg(k)
        out[cat] = out.get(cat, 0.0) + float(v)
    return out


def _make_particle_type_plot(ipc_pdg: Dict[int, float], hpp_pdg: Dict[int, float], title: str, outpath: str,
                             logy: bool, density: bool) -> None:
    cats = ["photon", "lepton", "neutrino", "pion", "kaon", "nucleon", "other"]
    m_ipc = _pdg_counts_to_categories(ipc_pdg)
    m_hpp = _pdg_counts_to_categories(hpp_pdg)
    y_ipc = np.array([m_ipc.get(c, 0.0) for c in cats], dtype=float)
    y_hpp = np.array([m_hpp.get(c, 0.0) for c in cats], dtype=float)
    if density:
        if y_ipc.sum() > 0:
            y_ipc = y_ipc / y_ipc.sum()
        if y_hpp.sum() > 0:
            y_hpp = y_hpp / y_hpp.sum()
    x = np.arange(len(cats))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    ax.bar(x - width / 2, y_ipc, width=width, color=color_for_source("IPC"), label="IPC")
    ax.bar(x + width / 2, y_hpp, width=width, color=color_for_source("HPP"), label="HPP")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylabel("Fraction" if density else "Counts")
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    #fig.savefig(outpath, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outpath)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Overlay IPC vs HPP distributions (energy, pT, pz) per collider. "
            "IPC: provide directories with pair files; HPP: provide merged edm4hep ROOT files."
        )
    )

    # IPC directories
    ap.add_argument("--ipc-C3-250-PS1", dest="ipc_C3_250_PS1", default="", help="IPC directory for C3_250_PS1")
    ap.add_argument("--ipc-C3-250-PS2", dest="ipc_C3_250_PS2", default="", help="IPC directory for C3_250_PS2")
    ap.add_argument("--ipc-C3-550-PS1", dest="ipc_C3_550_PS1", default="", help="IPC directory for C3_550_PS1")
    ap.add_argument("--ipc-C3-550-PS2", dest="ipc_C3_550_PS2", default="", help="IPC directory for C3_550_PS2")
    ap.add_argument("--ipc-pattern", default="*pairs*.dat", help="Glob pattern for IPC files within dirs (default: *pairs*.dat)")

    # HPP ROOT files
    ap.add_argument("--hpp-C3-250-PS1", dest="hpp_C3_250_PS1", default="", help="HPP edm4hep ROOT for C3_250_PS1")
    ap.add_argument("--hpp-C3-250-PS2", dest="hpp_C3_250_PS2", default="", help="HPP edm4hep ROOT for C3_250_PS2")
    ap.add_argument("--hpp-C3-550-PS1", dest="hpp_C3_550_PS1", default="", help="HPP edm4hep ROOT for C3_550_PS1")
    ap.add_argument("--hpp-C3-550-PS2", dest="hpp_C3_550_PS2", default="", help="HPP edm4hep ROOT for C3_550_PS2")
    ap.add_argument("--hpp-final-state", action="store_true", help="Only use MC particles with generatorStatus==1 (final state)")

    # Plotting / binning
    ap.add_argument("--bins", type=int, default=200, help="Histogram bins (default: 200)")
    ap.add_argument("--density", action="store_true", help="Normalize histograms to unit area")
    ap.add_argument("--logy", action="store_true", help="Logarithmic Y scale")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"], help="Output format")
    ap.add_argument("--out-dir", default=".", help="Output directory (default: current)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2), help="Max worker threads for IPC reading")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Logging level")
    ap.add_argument("--ipc-max-files", type=int, default=None, help="Max IPC files to read per directory (for speed)")

    # Optional manual ranges
    ap.add_argument("--energy-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"), help="Energy range in GeV")
    ap.add_argument("--pt-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"), help="pT range in GeV")
    ap.add_argument("--pz-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"), help="pz range in GeV")
    # Optional axis max caps (override upper x-limit only)
    ap.add_argument("--energy-xmax", type=float, default=None, help="Override max x-axis for energy plot")
    ap.add_argument("--pt-xmax", type=float, default=None, help="Override max x-axis for pT plot")
    ap.add_argument("--pz-xmax", type=float, default=None, help="Override max x-axis for pz plot")
    ap.add_argument("--energy-xmin", type=float, default=None, help="Override min x-axis for energy plot")
    ap.add_argument("--pt-xmin", type=float, default=None, help="Override min x-axis for pT plot")
    ap.add_argument("--pz-xmin", type=float, default=None, help="Override min x-axis for pz plot")
    # Optional y minimums
    ap.add_argument("--energy-ymin", type=float, default=None, help="Override min y-axis for energy plot")
    ap.add_argument("--pt-ymin", type=float, default=None, help="Override min y-axis for pT plot")
    ap.add_argument("--pz-ymin", type=float, default=None, help="Override min y-axis for pz plot")
    # PDG plotting controls
    ap.add_argument("--pdg-xmin", type=float, default=None, help="Override min x-axis for PDG plots")
    ap.add_argument("--pdg-xmax", type=float, default=None, help="Override max x-axis for PDG plots")
    # Save/load histograms
    ap.add_argument("--save-npz", action="store_true", help="Save computed histograms as NPZ files")
    ap.add_argument("--load-npz", action="store_true", help="Load histograms from NPZ instead of recomputing where available")
    ap.add_argument("--npz-dir", default=None, help="Directory to load NPZ files from (default: --out-dir)")

    return ap.parse_args()


def _collect_labels(args: argparse.Namespace) -> List[ColliderKey]:
    labels = ["C3_250_PS1", "C3_250_PS2", "C3_550_PS1", "C3_550_PS2"]
    available = []
    for lab in labels:
        ipc = getattr(args, f"ipc_{lab}")
        hpp = getattr(args, f"hpp_{lab}")
        if ipc or hpp:
            available.append(lab)
        elif args.load_npz:
            npz_dir = args.npz_dir or args.out_dir
            npz_path = os.path.join(npz_dir, f"{lab}.hists.npz")
            if os.path.isfile(npz_path):
                available.append(lab)
    return available


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    _ensure_outdir(args.out_dir)

    labels = _collect_labels(args)
    if not labels:
        raise SystemExit("No colliders provided. Use --ipc-*/--hpp-* arguments.")

    color_group_map: Dict[str, int] = {}
    group_name_to_idx: Dict[str, int] = {}
    next_color_idx = 0
    for lab in labels:
        if "PS1" in lab:
            group = "PS1"
        elif "PS2" in lab:
            group = "PS2"
        else:
            group = lab
        if group not in group_name_to_idx:
            group_name_to_idx[group] = next_color_idx
            next_color_idx += 1
        color_group_map[lab] = group_name_to_idx[group]
    color_total = max(1, len(group_name_to_idx))

    def _color_idx(label: str) -> int:
        return color_group_map.get(label, 0)

    # Hold per-label hists for combined overlays
    overlay = {"energy": [], "pt": [], "pz": []}  # list of tuples (source, label, x, y)
    overlay_pdg: List[Tuple[str, str, Dict[int, float]]] = []  # (source, label, pdg_counts)
    gmin = {"energy": np.inf, "pt": np.inf, "pz": np.inf}
    gmax = {"energy": -np.inf, "pt": -np.inf, "pz": -np.inf}
    avg_records: List[Dict[str, Any]] = []

    for label in labels:
        ipc_dir = getattr(args, f"ipc_{label}")
        hpp_file = getattr(args, f"hpp_{label}")

        # Determine ranges: prefer user-provided, else auto from data
        e_lo, e_hi = (args.energy_range if args.energy_range is not None else (None, None))
        pt_lo, pt_hi = (args.pt_range if args.pt_range is not None else (None, None))
        pz_lo, pz_hi = (args.pz_range if args.pz_range is not None else (None, None))

        # Auto-range pass if needed
        if None in (e_lo, e_hi) or None in (pt_lo, pt_hi) or None in (pz_lo, pz_hi):
            # Initialize with NaNs and expand with any available source (IPC/HPP)
            e_mm = [np.inf, -np.inf]
            pt_mm = [np.inf, -np.inf]
            pz_mm = [np.inf, -np.inf]

            if ipc_dir:
                e0, e1, pt0, pt1, pz0, pz1 = _read_ipc_pass_minmax(ipc_dir, args.ipc_pattern, max_files=args.ipc_max_files)
                e_mm[0], e_mm[1] = min(e_mm[0], e0), max(e_mm[1], e1)
                pt_mm[0], pt_mm[1] = min(pt_mm[0], pt0), max(pt_mm[1], pt1)
                pz_mm[0], pz_mm[1] = min(pz_mm[0], pz0), max(pz_mm[1], pz1)
            if hpp_file and os.path.isfile(hpp_file):
                h_e0, h_e1, h_pt0, h_pt1, h_pz0, h_pz1 = _hpp_pass_minmax(hpp_file, args.hpp_final_state)
                e_mm[0], e_mm[1] = min(e_mm[0], h_e0), max(e_mm[1], h_e1)
                pt_mm[0], pt_mm[1] = min(pt_mm[0], h_pt0), max(pt_mm[1], h_pt1)
                pz_mm[0], pz_mm[1] = min(pz_mm[0], h_pz0), max(pz_mm[1], h_pz1)

            if None in (e_lo, e_hi):
                e_lo, e_hi = _pad_range(e_mm[0], e_mm[1])
            if None in (pt_lo, pt_hi):
                pt_lo, pt_hi = _pad_range(pt_mm[0], pt_mm[1])
            if None in (pz_lo, pz_hi):
                pz_lo, pz_hi = _pad_range(pz_mm[0], pz_mm[1])

        ranges = {
            "energy": (float(e_lo), float(e_hi)),
            "pt": (float(pt_lo), float(pt_hi)),
            "pz": (float(pz_lo), float(pz_hi)),
        }

        # Compute or load histograms
        ipc_hists = {
            "energy": (np.array([]), np.array([])),
            "pt": (np.array([]), np.array([])),
            "pz": (np.array([]), np.array([])),
            "pdg_counts": {},
            "total_particles": 0.0,
            "bunch_crossings": 0,
        }
        hpp_hists = {
            "energy": (np.array([]), np.array([])),
            "pt": (np.array([]), np.array([])),
            "pz": (np.array([]), np.array([])),
            "pdg_counts": {},
            "total_particles": 0.0,
            "total_events": 0,
        }

        loaded_npz = False
        if args.load_npz:
            npz_dir = args.npz_dir or args.out_dir
            npz_path = os.path.join(npz_dir, f"{label}.hists.npz")
            if os.path.isfile(npz_path):
                try:
                    data = np.load(npz_path, allow_pickle=True)
                    ipc_pdg_np = data.get("ipc_pdg", None)
                    ipc_total_particles_np = data.get("ipc_total_particles", None)
                    ipc_bx_np = data.get("ipc_bunch_crossings", None)
                    ipc_hists = {
                        "energy": (data.get("energy_ipc_x", np.array([])), data.get("energy_ipc_y", np.array([]))),
                        "pt": (data.get("pt_ipc_x", np.array([])), data.get("pt_ipc_y", np.array([]))),
                        "pz": (data.get("pz_ipc_x", np.array([])), data.get("pz_ipc_y", np.array([]))),
                        "pdg_counts": ipc_pdg_np.item() if ipc_pdg_np is not None else {},
                        "total_particles": float(ipc_total_particles_np.item()) if ipc_total_particles_np is not None else 0.0,
                        "bunch_crossings": int(ipc_bx_np.item()) if ipc_bx_np is not None else 0,
                    }
                    hpp_pdg_np = data.get("hpp_pdg", None)
                    hpp_total_particles_np = data.get("hpp_total_particles", None)
                    hpp_events_np = data.get("hpp_total_events", None)
                    hpp_hists = {
                        "energy": (data.get("energy_hpp_x", np.array([])), data.get("energy_hpp_y", np.array([]))),
                        "pt": (data.get("pt_hpp_x", np.array([])), data.get("pt_hpp_y", np.array([]))),
                        "pz": (data.get("pz_hpp_x", np.array([])), data.get("pz_hpp_y", np.array([]))),
                        "pdg_counts": hpp_pdg_np.item() if hpp_pdg_np is not None else {},
                        "total_particles": float(hpp_total_particles_np.item()) if hpp_total_particles_np is not None else 0.0,
                        "total_events": int(hpp_events_np.item()) if hpp_events_np is not None else 0,
                    }
                    loaded_npz = True
                    logging.info("Loaded histograms from %s", npz_path)
                except Exception as exc:
                    logging.warning("Failed to load NPZ %s: %s; recomputing instead", npz_path, exc)

        if not loaded_npz:
            if ipc_dir:
                if os.path.isdir(ipc_dir):
                    ipc_hists = _fill_ipc_hists(
                        ipc_dir,
                        args.ipc_pattern,
                        args.bins,
                        ranges,
                        args.density,
                        max_workers=max(1, args.workers),
                        max_files=args.ipc_max_files,
                    )
                else:
                    logging.warning("IPC directory %s does not exist; skipping", ipc_dir)
            if hpp_file:
                if os.path.isfile(hpp_file):
                    hpp_hists = _fill_hpp_hists(
                        hpp_file,
                        args.bins,
                        ranges,
                        args.hpp_final_state,
                        args.density,
                    )
                else:
                    logging.warning("HPP file %s does not exist; skipping", hpp_file)

        ipc_total_particles = float(ipc_hists.get("total_particles", 0.0) or 0.0)
        ipc_bx = int(ipc_hists.get("bunch_crossings", 0) or 0)
        hpp_total_particles = float(hpp_hists.get("total_particles", 0.0) or 0.0)
        hpp_total_events = int(hpp_hists.get("total_events", 0) or 0)
        avg_records.append({
            "label": label,
            "ipc_total_particles": ipc_total_particles,
            "ipc_bunch_crossings": ipc_bx,
            "hpp_total_particles": hpp_total_particles,
            "hpp_total_events": hpp_total_events,
            "hpp_events_per_bx": None,
        })

        # Plot
        nice_label = label.replace("_", "-")
        ls = linestyle_for_label(label)
        out_energy = os.path.join(args.out_dir, f"{label}.energy.{args.format}")
        out_pt = os.path.join(args.out_dir, f"{label}.pt.{args.format}")
        out_pz = os.path.join(args.out_dir, f"{label}.pz.{args.format}")

        _make_plot(
            ipc_hists["energy"], hpp_hists["energy"], xlabel="Energy [GeV]", title=f"{nice_label} Energy",
            outpath=out_energy, logy=args.logy, density=args.density,
            ymin=args.energy_ymin, xmin=args.energy_xmin, xmax=args.energy_xmax, linestyle=ls)
        _make_plot(
            ipc_hists["pt"], hpp_hists["pt"], xlabel=r"$p_\mathrm{T}$ [GeV]", title=f"{nice_label} pT",
            outpath=out_pt, logy=args.logy, density=args.density,
            ymin=args.pt_ymin, xmin=args.pt_xmin, xmax=args.pt_xmax, linestyle=ls)
        _make_plot(
            ipc_hists["pz"], hpp_hists["pz"], xlabel=r"$p_z$ [GeV]", title=f"{nice_label} pz",
            outpath=out_pz, logy=args.logy, density=args.density,
            ymin=args.pz_ymin, xmin=args.pz_xmin, xmax=args.pz_xmax, linestyle=ls)

        # Per-collider PDG plot (if any counts present)
        pdg_ipc = ipc_hists.get("pdg_counts", {}) or {}
        pdg_hpp = hpp_hists.get("pdg_counts", {}) or {}
        if pdg_ipc or pdg_hpp:
            pdg_out = os.path.join(args.out_dir, f"{label}.pdg.{args.format}")
            _make_pdg_plot_with_limits(
                pdg_ipc, pdg_hpp, title=f"{nice_label} PDG", outpath=pdg_out, logy=args.logy,
                density=args.density, xmin=args.pdg_xmin, xmax=args.pdg_xmax, linestyle=ls)
            # Also particle-type summary plot
            ptype_out = os.path.join(args.out_dir, f"{label}.particle_type.{args.format}")
            _make_particle_type_plot(pdg_ipc, pdg_hpp, title=f"{nice_label} Particle types", outpath=ptype_out, logy=args.logy, density=args.density)

        # Save NPZ per collider if requested
        if args.save_npz:
            base = os.path.join(args.out_dir, f"{label}.hists.npz")
            np.savez(
                base,
                energy_ipc_x=ipc_hists["energy"][0], energy_ipc_y=ipc_hists["energy"][1],
                energy_hpp_x=hpp_hists["energy"][0], energy_hpp_y=hpp_hists["energy"][1],
                pt_ipc_x=ipc_hists["pt"][0], pt_ipc_y=ipc_hists["pt"][1],
                pt_hpp_x=hpp_hists["pt"][0], pt_hpp_y=hpp_hists["pt"][1],
                pz_ipc_x=ipc_hists["pz"][0], pz_ipc_y=ipc_hists["pz"][1],
                pz_hpp_x=hpp_hists["pz"][0], pz_hpp_y=hpp_hists["pz"][1],
                density=args.density,
                ipc_pdg=np.array(ipc_hists.get("pdg_counts", {}), dtype=object),
                hpp_pdg=np.array(hpp_hists.get("pdg_counts", {}), dtype=object),
                ipc_total_particles=np.array(ipc_hists.get("total_particles", 0.0), dtype=float),
                ipc_bunch_crossings=np.array(ipc_hists.get("bunch_crossings", 0), dtype=int),
                hpp_total_particles=np.array(hpp_hists.get("total_particles", 0.0), dtype=float),
                hpp_total_events=np.array(hpp_hists.get("total_events", 0), dtype=int),
            )

        # Accumulate for combined overlays
        for key, h in (("energy", ipc_hists["energy"]), ("pt", ipc_hists["pt"]), ("pz", ipc_hists["pz"])):
            x, y = h
            if x.size:
                overlay[key].append(("IPC", label, x, y))
                gmin[key] = min(gmin[key], float(np.min(x)))
                gmax[key] = max(gmax[key], float(np.max(x)))
        for key, h in (("energy", hpp_hists["energy"]), ("pt", hpp_hists["pt"]), ("pz", hpp_hists["pz"])):
            x, y = h
            if x.size:
                overlay[key].append(("HPP", label, x, y))
                gmin[key] = min(gmin[key], float(np.min(x)))
                gmax[key] = max(gmax[key], float(np.max(x)))
        # PDG overlay accumulation
        if pdg_ipc:
            overlay_pdg.append(("IPC", label, pdg_ipc))
        if pdg_hpp:
            overlay_pdg.append(("HPP", label, pdg_hpp))

    # Combined overlays across colliders: Energy, pT, pz
    for key, xlabel, stem in (("energy", "Energy [GeV]", "energy"), ("pt", r"$p_\mathrm{T}$ [GeV]", "pt"), ("pz", r"$p_z$ [GeV]", "pz")):
        if not overlay[key]:
            continue
        fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
        for src, lab, x, y in overlay[key]:
            idx = _color_idx(lab)
            color = color_for_source(src, idx, color_total)
            line_style = linestyle_for_label(lab)
            ax.plot(x, y, color=color, lw=1.4, label=f"{lab.replace('_','-')} ({src})", linestyle=line_style)
        # Axis and style
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized counts" if args.density else "Counts")
        # Base limits from data
        if np.isfinite(gmin[key]) and np.isfinite(gmax[key]) and gmax[key] > gmin[key]:
            left, right = gmin[key], gmax[key]
        else:
            left, right = ax.get_xlim()
        # Apply optional min/max caps
        if key == "energy":
            if args.energy_xmin is not None and np.isfinite(args.energy_xmin):
                left = float(args.energy_xmin)
            if args.energy_xmax is not None and np.isfinite(args.energy_xmax):
                right = float(args.energy_xmax)
        elif key == "pt":
            if args.pt_xmin is not None and np.isfinite(args.pt_xmin):
                left = float(args.pt_xmin)
            if args.pt_xmax is not None and np.isfinite(args.pt_xmax):
                right = float(args.pt_xmax)
        elif key == "pz":
            if args.pz_xmin is not None and np.isfinite(args.pz_xmin):
                left = float(args.pz_xmin)
            if args.pz_xmax is not None and np.isfinite(args.pz_xmax):
                right = float(args.pz_xmax)
        if right > left:
            ax.set_xlim(left, right)
        if args.logy:
            ax.set_yscale("log")
        # Optional y-min caps for combined overlays
        if key == "energy" and args.energy_ymin is not None and np.isfinite(args.energy_ymin):
            yl = ax.get_ylim()
            ax.set_ylim(bottom=float(args.energy_ymin), top=yl[1])
        if key == "pt" and args.pt_ymin is not None and np.isfinite(args.pt_ymin):
            yl = ax.get_ylim()
            ax.set_ylim(bottom=float(args.pt_ymin), top=yl[1])
        if key == "pz" and args.pz_ymin is not None and np.isfinite(args.pz_ymin):
            yl = ax.get_ylim()
            ax.set_ylim(bottom=float(args.pz_ymin), top=yl[1])
        ax.grid(alpha=0.25)
        # Grouped legends: one box for IPC, one for HPP
        handles_all, labels_all = ax.get_legend_handles_labels()
        group_handles = {"IPC": [], "HPP": []}
        group_labels = {"IPC": [], "HPP": []}
        for h, l in zip(handles_all, labels_all):
            if l.endswith("(IPC)"):
                group_handles["IPC"].append(h)
                group_labels["IPC"].append(l.replace(" (IPC)", ""))
            elif l.endswith("(HPP)"):
                group_handles["HPP"].append(h)
                group_labels["HPP"].append(l.replace(" (HPP)", ""))
        if group_handles["IPC"]:
            leg_ipc = ax.legend(group_handles["IPC"], group_labels["IPC"], title="IPC", ncol=1,
                                fontsize=10, title_fontsize=11, frameon=True,
                                loc='upper left', bbox_to_anchor=(0.05, 0.98), borderaxespad=0.)
            ax.add_artist(leg_ipc)
        if group_handles["HPP"]:
            if key == "pz":
                # For pz overlay, place HPP legend further to the right to avoid overlap
                leg_hpp = ax.legend(group_handles["HPP"], group_labels["HPP"], title="HPP", ncol=1,
                                    fontsize=10, title_fontsize=11, frameon=True,
                                    loc='upper right', bbox_to_anchor=(0.98, 0.98), borderaxespad=0.)
            else:
                leg_hpp = ax.legend(group_handles["HPP"], group_labels["HPP"], title="HPP", ncol=1,
                                    fontsize=10, title_fontsize=11, frameon=True,
                                    loc='upper left', bbox_to_anchor=(0.35, 0.98), borderaxespad=0.)
            ax.add_artist(leg_hpp)
        fig.tight_layout()
        out_all = os.path.join(args.out_dir, f"all.{stem}.{args.format}")
        #fig.savefig(out_all, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_all)
        # Save combined overlay NPZ if requested
        if args.save_npz:
            # Build stacked arrays for easy reuse
            entries = overlay[key]
            sources = np.array([src for src, _, _, _ in entries], dtype=object)
            labels_arr = np.array([lab for _, lab, _, _ in entries], dtype=object)
            # Note: x-grids may differ per curve; save as object arrays
            x_list = [x for _, _, x, _ in entries]
            y_list = [y for _, _, _, y in entries]
            try:
                np.savez(
                    os.path.join(args.out_dir, f"all.{stem}.hists.npz"),
                    sources=sources, labels=labels_arr, x_list=np.array(x_list, dtype=object), y_list=np.array(y_list, dtype=object),
                    density=args.density,
                )
            except Exception as exc:
                logging.warning("Failed to save combined NPZ for %s: %s", stem, exc)
        plt.close(fig)

    # Combined PDG overlay using in-memory counts; fallback to NPZ only if in-memory is empty
    pdg_overlay_map: Dict[str, Dict[str, Dict[int, float]]] = {lab: {"IPC": {}, "HPP": {}} for lab in labels}
    for src, lab, pdg_counts in overlay_pdg:
        pdg_overlay_map[lab][src] = pdg_counts
    # If no in-memory PDG (e.g., loaded solely from NPZ without counts), try to load from NPZ
    if all((not pdg_overlay_map[lab]["IPC"]) and (not pdg_overlay_map[lab]["HPP"]) for lab in labels):
        pdg_npz_dir = args.npz_dir or args.out_dir
        for lab in labels:
            npz_path = os.path.join(pdg_npz_dir, f"{lab}.hists.npz")
            try:
                if os.path.isfile(npz_path):
                    d = np.load(npz_path, allow_pickle=True)
                    pdg_overlay_map[lab]["IPC"] = d.get("ipc_pdg", np.array([])).item() if d.get("ipc_pdg", None) is not None else {}
                    pdg_overlay_map[lab]["HPP"] = d.get("hpp_pdg", np.array([])).item() if d.get("hpp_pdg", None) is not None else {}
            except Exception:
                pass

    # Prepare entries list
    pdg_overlay_entries = [(lab, pdg_overlay_map[lab]["IPC"], pdg_overlay_map[lab]["HPP"]) for lab in labels
                           if (pdg_overlay_map[lab]["IPC"] or pdg_overlay_map[lab]["HPP"]) ]

    if pdg_overlay_entries:
        fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
        # Build union of PDG IDs across all
        all_ids = sorted(set().union(*[set(map(int, list(ipc_d.keys()) + list(hpp_d.keys()))) for _, ipc_d, hpp_d in pdg_overlay_entries]))
        # Plot IPC and HPP per collider using color-blind-friendly variants
        for lab, ipc_pdg, hpp_pdg in pdg_overlay_entries:
            idx = _color_idx(lab)
            col_ipc = color_for_source("IPC", idx, color_total)
            col_hpp = color_for_source("HPP", idx, color_total)
            line_style = linestyle_for_label(lab)
            # Convert dicts to aligned arrays across all_ids, fill zeros
            y_ipc = np.array([float(ipc_pdg.get(pid, 0.0)) for pid in all_ids], dtype=float)
            y_hpp = np.array([float(hpp_pdg.get(pid, 0.0)) for pid in all_ids], dtype=float)
            if args.density and y_ipc.sum() > 0:
                y_ipc = y_ipc / y_ipc.sum()
            if args.density and y_hpp.sum() > 0:
                y_hpp = y_hpp / y_hpp.sum()
            ax.plot(all_ids, y_ipc, color=col_ipc, lw=1.4, marker="o", ms=3,
                    label=f"{lab.replace('_','-')} (IPC)", linestyle=line_style)
            ax.plot(all_ids, y_hpp, color=col_hpp, lw=1.4, marker="o", ms=3,
                    label=f"{lab.replace('_','-')} (HPP)", linestyle=line_style)
        ax.set_xlabel("PDG ID")
        ax.set_ylabel("Fraction" if args.density else "Counts")
        if args.logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
        # Apply user x-range for PDG
        left, right = ax.get_xlim()
        if args.pdg_xmin is not None and np.isfinite(args.pdg_xmin):
            left = float(args.pdg_xmin)
        if args.pdg_xmax is not None and np.isfinite(args.pdg_xmax):
            right = float(args.pdg_xmax)
        if right > left:
            ax.set_xlim(left, right)
        # Ticks: integers within the range
        try:
            xticks = np.arange(int(np.ceil(left)), int(np.floor(right)) + 1)
            if xticks.size <= 50:
                ax.set_xticks(xticks)
                ax.set_xticklabels([str(i) for i in xticks], rotation=45, ha="right")
        except Exception:
            # fallback to union ticks
            ax.set_xticks(all_ids)
            ax.set_xticklabels([str(i) for i in all_ids], rotation=45, ha="right")
        # Group legends like other overlays
        handles_all, labels_all = ax.get_legend_handles_labels()
        group_handles = {"IPC": [], "HPP": []}
        group_labels = {"IPC": [], "HPP": []}
        for h, l in zip(handles_all, labels_all):
            if l.endswith("(IPC)"):
                group_handles["IPC"].append(h)
                group_labels["IPC"].append(l.replace(" (IPC)", ""))
            elif l.endswith("(HPP)"):
                group_handles["HPP"].append(h)
                group_labels["HPP"].append(l.replace(" (HPP)", ""))
        if group_handles["IPC"]:
            leg_ipc = ax.legend(group_handles["IPC"], group_labels["IPC"], title="IPC", ncol=1,
                                fontsize=10, title_fontsize=11, frameon=True,
                                loc='upper left', bbox_to_anchor=(0.05, 0.98), borderaxespad=0.)
            ax.add_artist(leg_ipc)
        if group_handles["HPP"]:
            leg_hpp = ax.legend(group_handles["HPP"], group_labels["HPP"], title="HPP", ncol=1,
                                fontsize=10, title_fontsize=11, frameon=True,
                                loc='upper left', bbox_to_anchor=(0.35, 0.98), borderaxespad=0.)
            ax.add_artist(leg_hpp)
        fig.tight_layout()
        out_pdg_all = os.path.join(args.out_dir, f"all.pdg.{args.format}")
        #fig.savefig(out_pdg_all, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_pdg_all)
        if args.save_npz:
            try:
                np.savez(
                    os.path.join(args.out_dir, f"all.pdg.hists.npz"),
                    labels=np.array([lab for lab, _, _ in pdg_overlay_entries], dtype=object),
                    ipc_dicts=np.array([e for _, e, _ in pdg_overlay_entries], dtype=object),
                    hpp_dicts=np.array([e for _, _, e in pdg_overlay_entries], dtype=object),
                    density=args.density,
                )
            except Exception as exc:
                logging.warning("Failed to save combined PDG NPZ: %s", exc)
        plt.close(fig)

    # Combined particle-type overlay (bars) across colliders
    # Build combined IPC/HPP PDG dicts per collider
    if pdg_overlay_entries:
        fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
        # For the combined plot, drop the "other" category
        cats = ["photon", "lepton", "neutrino", "pion", "kaon", "nucleon"]
        x = np.arange(len(cats))
        N = max(1, len(pdg_overlay_entries))
        barw = 0.8 / (N * 2.0)
        # Plot paired bars (IPC/HPP) per collider without overlap
        for i, (lab, ipc_pdg, hpp_pdg) in enumerate(pdg_overlay_entries):
            idx = _color_idx(lab)
            col_ipc = color_for_source("IPC", idx, color_total)
            col_hpp = color_for_source("HPP", idx, color_total)
            hatch_style = "//" if "C3_550" in lab else None
            m_ipc = _pdg_counts_to_categories(ipc_pdg)
            m_hpp = _pdg_counts_to_categories(hpp_pdg)
            y_ipc = np.array([m_ipc.get(c, 0.0) for c in cats], dtype=float)
            y_hpp = np.array([m_hpp.get(c, 0.0) for c in cats], dtype=float)
            if args.density:
                if y_ipc.sum() > 0:
                    y_ipc = y_ipc / y_ipc.sum()
                if y_hpp.sum() > 0:
                    y_hpp = y_hpp / y_hpp.sum()
            center_offset = (i - (N - 1) / 2.0) * (2.0 * barw)
            ipc_pos = x + center_offset - barw * 0.5
            hpp_pos = x + center_offset + barw * 0.5
            ax.bar(ipc_pos, y_ipc, width=barw, color=col_ipc, alpha=0.85, edgecolor='black', linewidth=0.3,
                   hatch=hatch_style,
                   label=f"{lab.replace('_','-')} (IPC)")
            ax.bar(hpp_pos, y_hpp, width=barw, color=col_hpp, alpha=0.85, edgecolor='black', linewidth=0.3,
                   hatch=hatch_style,
                   label=f"{lab.replace('_','-')} (HPP)")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=30, ha="right")
        ax.set_ylabel("Fraction" if args.density else "Counts")
        if args.logy:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        # Group legends
        handles_all, labels_all = ax.get_legend_handles_labels()
        group_handles = {"IPC": [], "HPP": []}
        group_labels = {"IPC": [], "HPP": []}
        for h, l in zip(handles_all, labels_all):
            if l.endswith("(IPC)"):
                group_handles["IPC"].append(h)
                group_labels["IPC"].append(l.replace(" (IPC)", ""))
            elif l.endswith("(HPP)"):
                group_handles["HPP"].append(h)
                group_labels["HPP"].append(l.replace(" (HPP)", ""))
        if group_handles["IPC"]:
            leg_ipc = ax.legend(group_handles["IPC"], group_labels["IPC"], title="IPC", ncol=1,
                                fontsize=10, title_fontsize=11, frameon=True,
                                loc='upper left', bbox_to_anchor=(0.35, 0.98), borderaxespad=0.)
            ax.add_artist(leg_ipc)
        if group_handles["HPP"]:
            leg_hpp = ax.legend(group_handles["HPP"], group_labels["HPP"], title="HPP", ncol=1,
                                fontsize=10, title_fontsize=11, frameon=True,
                                loc='upper right', bbox_to_anchor=(0.98, 0.98), borderaxespad=0.)
            ax.add_artist(leg_hpp)
        fig.tight_layout()
        out_ptype_all = os.path.join(args.out_dir, f"all.particle_type.{args.format}")
        #fig.savefig(out_ptype_all, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_ptype_all)
        plt.close(fig)

    if avg_records:
        for record in avg_records:
            if record["hpp_total_events"] > 0:
                while True:
                    try:
                        prompt = f"Enter expected HPP events per bunch crossing for {record['label'].replace('_', '-')} (non-negative number): "
                        entry = input(prompt).strip()
                        events_per_bx = float(entry)
                        if events_per_bx < 0:
                            print("Please enter a non-negative number.")
                            continue
                        record["hpp_events_per_bx"] = events_per_bx
                        break
                    except ValueError:
                        print("Invalid input. Please enter a numeric value.")
            else:
                record["hpp_events_per_bx"] = None

        print("\nAverage number of particles per bunch crossing:")
        for record in avg_records:
            label_readable = record["label"].replace('_', '-')
            ipc_crossings = record["ipc_bunch_crossings"]
            if ipc_crossings > 0:
                ipc_avg = record["ipc_total_particles"] / ipc_crossings
                print(f"  {label_readable} IPC: {ipc_avg:.3f} (from {record['ipc_total_particles']:.0f} particles across {ipc_crossings} bunch crossings)")
            else:
                print(f"  {label_readable} IPC: N/A (no IPC data)")
            if record["hpp_total_events"] > 0:
                events_per_bx = record.get("hpp_events_per_bx")
                particles_per_event = record["hpp_total_particles"] / record["hpp_total_events"] if record["hpp_total_events"] else 0.0
                if events_per_bx is not None:
                    hpp_avg = particles_per_event * events_per_bx
                    print(f"  {label_readable} HPP: {hpp_avg:.3f} (particles/event={particles_per_event:.3f}, events/bx={events_per_bx:.3f})")
                else:
                    print(f"  {label_readable} HPP: N/A (events per bunch crossing not provided)")
            else:
                print(f"  {label_readable} HPP: N/A (no HPP data)")


if __name__ == "__main__":
    main()

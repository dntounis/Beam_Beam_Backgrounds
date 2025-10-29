import os
import re
import argparse
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
from typing import Optional, Dict, List, Tuple

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


def discover_files(base_dir: str, pattern: str) -> List[Tuple[int, str]]:
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


def _find_tracker_energy_leaf(tree: uproot.TTree, branch: str) -> Optional[str]:
    """Return the tracker energy leaf name if present (expected: '<branch>.eDep')."""
    name = f"{branch}.eDep"
    try:
        _ = tree.keys(filter_name=name)
        return name
    except Exception:
        return None


def read_tracker_times(tree: uproot.TTree, branch: str, energy_threshold: float = 0.0) -> np.ndarray:
    """Read tracker hit times across all entries in the file.

    Applies non-zero position filter and returns a flattened 1D numpy array of times.
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
    hit_time = arrays[f"{branch}.time"]
    energy = arrays[energy_leaf] if energy_leaf is not None else None

    # we filter out hits where ALL coordinates are zero
    mask = ~((posx == 0) & (posy == 0) & (posz == 0))
    if energy is not None and energy_threshold > 0.0:
        mask = mask & (energy >= energy_threshold)

    if len(hit_time) == 0:
        return np.empty((0,), dtype=np.float64)

    filtered_times = ak.flatten(hit_time[mask], axis=None)
    times = ak.to_numpy(filtered_times).astype(np.float64, copy=False)
    return times


def read_calo_times(tree: uproot.TTree, branch: str, energy_threshold: float = 0.0) -> np.ndarray:
    """Read calorimeter hit times (all entries) using the first contribution per hit.

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
    energy = arrays[energy_leaf] if energy_leaf is not None else None

    # we filter out hits where ALL coordinates are zero
    hit_valid = ~((posx == 0) & (posy == 0) & (posz == 0)) & (beg < end)
    if energy is not None and energy_threshold > 0.0:
        first_energy = energy[beg]
        hit_valid = hit_valid & (first_energy >= energy_threshold)

    if len(posx) == 0:
        return np.empty((0,), dtype=np.float64)

    valid = hit_valid
    if not bool(ak.any(valid)):
        return np.empty((0,), dtype=np.float64)

    first_time = ctime[beg]
    filtered_times = ak.flatten(first_time[valid], axis=None)
    times = ak.to_numpy(filtered_times).astype(np.float64, copy=False)
    return times


def make_bins(lo: float, hi: float, base_binwidth: float) -> np.ndarray:
    nbins = int(round((hi - lo) / base_binwidth))
    return np.linspace(lo, hi, nbins + 1, dtype=np.float64)


def rebin_counts(counts: np.ndarray, edges: np.ndarray, new_binwidth: float, xmin: float, xmax: float) -> Tuple[np.ndarray, np.ndarray]:
    base_bw = edges[1] - edges[0]
    ratio = new_binwidth / base_bw
    if abs(round(ratio) - ratio) > 1e-6:
        raise ValueError(f"plot bin width {new_binwidth} is not a multiple of base bin width {base_bw}")
    ratio = int(round(ratio))
    i0 = max(0, int(np.floor((xmin - edges[0]) / base_bw)))
    i1 = min(counts.size, int(np.ceil((xmax - edges[0]) / base_bw)))
    counts_win = counts[i0:i1]
    pad = (-counts_win.size) % ratio
    if pad:
        counts_win = np.pad(counts_win, (0, pad), mode='constant')
    rebinned = counts_win.reshape(-1, ratio).sum(axis=1)
    edges_new = np.arange(edges[0] + i0 * base_bw, edges[0] + (i0 + rebinned.size * ratio) * base_bw + 1e-12, new_binwidth)
    return rebinned, edges_new


def plot_hist(branch: str,
              counts_ipc: np.ndarray,
              edges: np.ndarray,
              outdir: str,
              plot_xmin: float,
              plot_xmax: float,
              plot_binwidth: float,
              collider: str,
              detector_version: str,
              use_mplhep: bool = True,
              title_fs: int = 15,
              label_fs: int = 14,
              tick_fs: int = 11,
              legend_fs: int = 13,
              show_mean: bool = False,
              log_y: bool = False,
              counts_hpp: Optional[np.ndarray] = None,
              hpp_mu: Optional[float] = None) -> None:
    try:
        if use_mplhep:
            import mplhep as hep  # noqa: F401
            import mplhep as hep_style
            hep_style.style.use("CMS")
    except Exception:
        pass

    counts_reb, edges_reb = rebin_counts(counts_ipc, edges, plot_binwidth, plot_xmin, plot_xmax)
    bin_width = plot_binwidth
    centers = 0.5 * (edges_reb[:-1] + edges_reb[1:])

    def _fit_const(y_vals: np.ndarray) -> Tuple[float, float]:
        # Use the actual plot range instead of original histogram edges
        plot_min = plot_xmin
        plot_max = plot_xmax
        # Apply 5ns buffer from plot edges, but ensure we don't go outside the available data
        fit_min = max(plot_min + 5.0, centers[0] if len(centers) > 0 else plot_min)
        fit_max = min(plot_max - 5.0, centers[-1] if len(centers) > 0 else plot_max)
        mask = (centers >= fit_min) & (centers <= fit_max)
        y_fit = y_vals[mask]
        if not show_mean:
            return 0.0, 0.0
        if _HAS_SCIPY and y_fit.size > 0:
            def _const(x, a):
                return np.full_like(x, a)
            x_fit = centers[mask]
            try:
                popt, pcov = curve_fit(_const, x_fit, y_fit)
                perr = np.sqrt(np.diag(pcov)) if pcov.size else np.array([0.0])
                return float(popt[0]), float(perr[0])
            except Exception:
                pass
        if y_fit.size > 1:
            mean = float(np.mean(y_fit))
            stderr = float(np.std(y_fit, ddof=1) / np.sqrt(y_fit.size))
            return mean, stderr
        mean = float(np.mean(y_fit) if y_fit.size else 0.0)
        return mean, 0.0

    plt.figure(figsize=(8, 5))
    y_ipc = counts_reb / bin_width
    plt.step(centers, y_ipc, where="mid", color='blue', label='IPC')
    if show_mean:
        mean_ipc, stderr_ipc = _fit_const(counts_reb)
        plt.plot(centers, np.full_like(centers, mean_ipc) / bin_width, color='blue', linestyle='--', label='IPC mean')
        plt.fill_between(centers, (mean_ipc - stderr_ipc) / bin_width, (mean_ipc + stderr_ipc) / bin_width,
                         color='blue', alpha=0.15, label='IPC 68% CL')

    if counts_hpp is not None:
        counts_hpp_reb, _ = rebin_counts(counts_hpp, edges, plot_binwidth, plot_xmin, plot_xmax)
        y_hpp = counts_hpp_reb / bin_width
        plt.step(centers, y_hpp, where="mid", color='orange', label=('HPP' if hpp_mu is None else f'HPP (<N>={hpp_mu:g})'))
        if show_mean:
            mean_hpp, stderr_hpp = _fit_const(counts_hpp_reb)
            plt.plot(centers, np.full_like(centers, mean_hpp) / bin_width, color='orange', linestyle='--', label='HPP mean')
            plt.fill_between(centers, (mean_hpp - stderr_hpp) / bin_width, (mean_hpp + stderr_hpp) / bin_width,
                             color='orange', alpha=0.15, label='HPP 68% CL')
    plt.xlim(plot_xmin, plot_xmax)
    ax = plt.gca()
    # Ensure axis limits are strictly enforced
    ax.set_xlim(plot_xmin, plot_xmax)
    # Debug: verify axis limits are set correctly
    actual_xlim = ax.get_xlim()
    if abs(actual_xlim[1] - plot_xmax) > 1e-6:
        print(f"Warning: {branch} axis limit mismatch. Requested: {plot_xmax}, Actual: {actual_xlim[1]}")
    ax.tick_params(labelsize=tick_fs)
    plt.xlabel('Time (ns)', fontsize=label_fs)
    plt.ylabel(f'Number of hits/{bin_width:.2f} ns', fontsize=label_fs)
    if log_y:
        ax.set_yscale('log')
        # ensure a positive lower bound if zeros are present
        pos = y_ipc[y_ipc > 0]
        if pos.size:
            ymin = max(min(pos) * 0.5, 1e-12)
            ymax = ax.get_ylim()[1]
            ax.set_ylim(ymin, ymax)
    left_title = f"{collider} (1 bunch)"
    right_title = f"{detector_version} - {branch}"
    plt.title(left_title, fontsize=title_fs, loc='left')
    plt.title(right_title, fontsize=title_fs, loc='right')
    # Keep per-curve legend entries; no extra global mean text when overlaying
    plt.legend(fontsize=legend_fs)
    png = os.path.join(outdir, f"{collider}_one_BX_timing_{branch}.png")
    pdf = os.path.join(outdir, f"{collider}_one_BX_timing_{branch}.pdf")
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="EDM4hep timing histogrammer (one-BX average)")
    parser.add_argument("--base-dir", required=True, help="Directory containing EDM4hep ROOT files")
    parser.add_argument(
        "--pattern",
        default="ddsim_C3_250_PS1_v2_seed_*.edm4hep.root",
        help="Glob pattern (under base-dir) to discover files (default: ddsim_C3_250_PS1_v2_seed_*.edm4hep.root)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds to include (space-separated)")
    parser.add_argument("--out", default="timing_one_BX_histograms.npz", help="Output npz file for histograms")
    parser.add_argument("--outdir", default=".", help="Directory to write PNG/PDF plots")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files (after seed filtering)")
    parser.add_argument("--time-min", type=float, default=0.0, help="X-axis minimum for plotting and base histogram")
    parser.add_argument("--time-max", type=float, default=800.0, help="X-axis maximum for plotting and base histogram")
    parser.add_argument("--base-binwidth", type=float, default=0.25, help="Base bin width (ns) for accumulation")
    parser.add_argument("--plot-binwidth", type=float, default=10.0, help="Bin width (ns) for plotting and fit")
    parser.add_argument("--collider", default="C3 250 PS1", help="Collider scenario label for title")
    parser.add_argument("--detector", default="SiD_o2_v04", help="Detector version label for title")
    parser.add_argument("--load-hist", default=None, help="Load histograms from an NPZ (skip reading ROOT files)")
    parser.add_argument("--use-cache", action='store_true', help="If --out exists and matches config, reuse it")
    parser.add_argument("--legend-fs", type=int, default=10, help="Legend font size for all plots")
    parser.add_argument("--show-mean", action='store_true', help="Compute and display mean with 68%% CL band")
    parser.add_argument("--logy", action='store_true', help="Enable log scale on y-axis")
    parser.add_argument("--hpp-file", default=None, help="Path to HPP edm4hep ROOT file (many events)")
    parser.add_argument("--hpp-mu", type=float, default=None, help="Expected mean number of HPP events per bunch crossing")
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
    # Maximum time for plotting per subdetector (defaults to --time-max if not specified)
    parser.add_argument("--tmax-vertex-barrel", type=float, default=None, help="Max time for SiVertexBarrelHits plots (default: use --time-max)")
    parser.add_argument("--tmax-vertex-endcap", type=float, default=None, help="Max time for SiVertexEndcapHits plots (default: use --time-max)")
    parser.add_argument("--tmax-tracker-barrel", type=float, default=None, help="Max time for SiTrackerBarrelHits plots (default: use --time-max)")
    parser.add_argument("--tmax-tracker-endcap", type=float, default=None, help="Max time for SiTrackerEndcapHits plots (default: use --time-max)")
    parser.add_argument("--tmax-tracker-forward", type=float, default=None, help="Max time for SiTrackerForwardHits plots (default: use --time-max)")
    parser.add_argument("--tmax-ecal-barrel", type=float, default=None, help="Max time for ECalBarrelHits plots (default: use --time-max)")
    parser.add_argument("--tmax-ecal-endcap", type=float, default=None, help="Max time for ECalEndcapHits plots (default: use --time-max)")
    parser.add_argument("--tmax-hcal-barrel", type=float, default=None, help="Max time for HCalBarrelHits plots (default: use --time-max)")
    parser.add_argument("--tmax-hcal-endcap", type=float, default=None, help="Max time for HCalEndcapHits plots (default: use --time-max)")
    parser.add_argument("--tmax-muon-barrel", type=float, default=None, help="Max time for MuonBarrelHits plots (default: use --time-max)")
    parser.add_argument("--tmax-muon-endcap", type=float, default=None, help="Max time for MuonEndcapHits plots (default: use --time-max)")
    parser.add_argument("--tmax-lumical", type=float, default=None, help="Max time for LumiCalHits plots (default: use --time-max)")
    parser.add_argument("--tmax-beamcal", type=float, default=None, help="Max time for BeamCalHits plots (default: use --time-max)")
    args = parser.parse_args()

    # Helper function to get subdetector-specific max time
    def get_tmax_for_branch(branch_name: str) -> float:
        """Get the appropriate time maximum for plotting based on branch name."""
        tmax_map = {
            "SiVertexBarrelHits": args.tmax_vertex_barrel,
            "SiVertexEndcapHits": args.tmax_vertex_endcap,
            "SiTrackerBarrelHits": args.tmax_tracker_barrel,
            "SiTrackerEndcapHits": args.tmax_tracker_endcap,
            "SiTrackerForwardHits": args.tmax_tracker_forward,
            "ECalBarrelHits": args.tmax_ecal_barrel,
            "ECalEndcapHits": args.tmax_ecal_endcap,
            "HCalBarrelHits": args.tmax_hcal_barrel,
            "HCalEndcapHits": args.tmax_hcal_endcap,
            "MuonBarrelHits": args.tmax_muon_barrel,
            "MuonEndcapHits": args.tmax_muon_endcap,
            "LumiCalHits": args.tmax_lumical,
            "BeamCalHits": args.tmax_beamcal,
        }
        # Return subdetector-specific value if set, otherwise fall back to global time-max
        return tmax_map.get(branch_name, args.time_max) or args.time_max

    # Calculate the maximum time needed across all subdetectors for data collection
    all_tmax_values = [args.time_max]  # Start with global value
    for branch in TRACKER_BRANCHES + CALO_BRANCHES:
        branch_tmax = get_tmax_for_branch(branch)
        all_tmax_values.append(branch_tmax)
    
    # Use the maximum of all tmax values for histogram creation to ensure we collect enough data
    histogram_time_max = max(all_tmax_values)
    if histogram_time_max > args.time_max:
        print(f"Info: Extending data collection range to {histogram_time_max}ns (from global {args.time_max}ns) to accommodate subdetector-specific ranges")

    # Option 1: load precomputed histograms (already per-BX scaled)
    if args.load_hist:
        with np.load(args.load_hist, allow_pickle=True) as npz:
            edges = npz["edges"]
            hist_counts = {b: npz[f"counts_{b}"] for b in TRACKER_BRANCHES + CALO_BRANCHES if f"counts_{b}" in npz}
            cached_files = list(npz.get("files", []))
        files = [(idx, path) for idx, path in enumerate(cached_files)]
        print(f"Loaded histograms from {args.load_hist}; skipping ROOT reads.")
    else:
        files = discover_files(args.base_dir, args.pattern)
        # filter by --seeds if provided
        if args.seeds is not None:
            seed_set = set(args.seeds)
            files = [fp for fp in files if fp[0] in seed_set]
            if not files:
                raise SystemExit("No files found for the requested --seeds.")
        if args.max_files is not None:
            files = files[: args.max_files]
        if not files:
            raise SystemExit("No files found. Check --base-dir, --pattern, and --seeds.")

        edges = make_bins(args.time_min, histogram_time_max, args.base_binwidth)
        hist_counts: dict[str, np.ndarray] = {}
        for b in TRACKER_BRANCHES + CALO_BRANCHES:
            hist_counts[b] = np.zeros(edges.size - 1, dtype=np.float64)

        # Reuse cached NPZ if compatible
        if args.use_cache and os.path.exists(args.out):
            try:
                with np.load(args.out, allow_pickle=True) as npz:
                    edges_cached = npz["edges"]
                    files_cached = list(npz.get("files", []))
                    if np.allclose(edges_cached, edges) and len(files_cached) == len(files) and all(
                        os.path.basename(a[1]) == os.path.basename(b) for a, b in zip(files, files_cached)
                    ):
                        print(f"Using cache from {args.out}")
                        for b in TRACKER_BRANCHES + CALO_BRANCHES:
                            key = f"counts_{b}"
                            if key in npz:
                                hist_counts[b] = npz[key]
                        files = [(i, p) for i, p in enumerate(files_cached)]
                        files_to_process = []
                    else:
                        files_to_process = files
            except Exception:
                files_to_process = files
        else:
            files_to_process = files

    # Process ROOT files if needed (no time offsets)
    files_processed = 0
    if not args.load_hist and files_to_process:
        for ifile, (seed, fp) in enumerate(files_to_process):
            print(f"Processing file {ifile} with seed = {seed}: {fp}")
            try:
                with uproot.open(fp) as f:
                    tree = f["events"]
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
                            t = read_tracker_times(tree, b, thr)
                            if t.size:
                                c, _ = np.histogram(t, bins=edges)
                                hist_counts[b] += c
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
                            t = read_calo_times(tree, b, thr)
                            if t.size:
                                c, _ = np.histogram(t, bins=edges)
                                hist_counts[b] += c
                        except KeyError:
                            continue
                files_processed += 1
            except Exception as e:
                print(f"Warning: failed to read {fp}: {e}")
                continue

    os.makedirs(args.outdir, exist_ok=True)

    # Scale by number of processed files to obtain per-BX results for IPC
    if not args.load_hist and files_to_process and files_processed > 0:
        for b in TRACKER_BRANCHES + CALO_BRANCHES:
            hist_counts[b] = hist_counts[b] / float(files_processed)

    # HPP processing: read single file with many events, normalize per BX using hpp_mu
    hpp_counts: Optional[Dict[str, np.ndarray]] = None
    num_hpp_events = 0
    if args.hpp_file is not None:
        print(f"Starting HPP background processing from: {args.hpp_file}")
        hpp_counts = {b: np.zeros(edges.size - 1, dtype=np.float64) for b in TRACKER_BRANCHES + CALO_BRANCHES}
        try:
            with uproot.open(args.hpp_file) as hf:
                htree = hf["events"]
                # Determine number of events
                num_hpp_events = htree.num_entries
                print(f"HPP file contains {num_hpp_events} events")
                # Iterate in chunks to control memory
                step = min(10000, num_hpp_events)  # events per chunk; larger chunks for better performance
                print(f"Processing HPP events in chunks of {step}...")
                start_time = time.time()
                for start in range(0, num_hpp_events, step):
                    stop = min(start + step, num_hpp_events)
                    elapsed = time.time() - start_time
                    rate = stop / elapsed if elapsed > 0 else 0
                    eta_sec = (num_hpp_events - stop) / rate if rate > 0 else 0
                    print(f"  Processing HPP events {start+1}-{stop} of {num_hpp_events} ({100*(stop)/num_hpp_events:.1f}%, {rate:.0f} evt/s, ETA: {eta_sec/60:.1f}min)")
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
                            # Load needed leaves for this range
                            names = [
                                f"{b}", f"{b}.position.x", f"{b}.position.y", f"{b}.position.z", f"{b}.time",
                            ]
                            energy_leaf = _find_tracker_energy_leaf(htree, b)
                            if energy_leaf is not None:
                                names.append(energy_leaf)
                            arr = htree.arrays(names, entry_start=start, entry_stop=stop, library="ak")
                            posx = ak.flatten(arr[f"{b}.position.x"])  # concat across events
                            posy = ak.flatten(arr[f"{b}.position.y"]) 
                            posz = ak.flatten(arr[f"{b}.position.z"]) 
                            hit_time = ak.flatten(arr[f"{b}.time"]) 
                            energy = ak.flatten(arr[energy_leaf]) if energy_leaf is not None else None
                            # we filter out hits where ALL coordinates are zero
                            mask = ~((posx == 0) & (posy == 0) & (posz == 0))
                            # mask = (posx != 0) & (posy != 0) & (posz != 0)
                            if energy is not None and thr > 0.0:
                                mask = mask & (energy >= thr)
                            if ak.sum(mask) > 0:
                                t = ak.to_numpy(hit_time[mask]).astype(np.float64, copy=False)
                                c, _ = np.histogram(t, bins=edges)
                                hpp_counts[b] += c
                        except KeyError:
                            continue  # Skip missing branches
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
                            # compute first-contribution time and energy per hit across events
                            # we filter out hits where ALL coordinates are zero
                            hit_valid = ~((posx == 0) & (posy == 0) & (posz == 0)) & (beg < end)
                            # hit_valid = (posx != 0) & (posy != 0) & (posz != 0) & (beg < end)
                            if energy is not None and thr > 0.0:
                                first_energy = energy[beg]
                                hit_valid = hit_valid & (first_energy >= thr)
                            if ak.sum(hit_valid) > 0:
                                first_time = ctime[beg]
                                t = ak.to_numpy(ak.flatten(first_time[hit_valid])).astype(np.float64, copy=False)
                                c, _ = np.histogram(t, bins=edges)
                                hpp_counts[b] += c
                        except KeyError:
                            continue  # Skip missing branches
                print(f"Completed HPP processing of {num_hpp_events} events")
        except Exception as e:
            print(f"Warning: failed to read HPP file {args.hpp_file}: {e}")
            hpp_counts = None

    # Normalize HPP per BX using number of events and provided mu
    if hpp_counts is not None:
        print("Normalizing HPP histograms per bunch crossing...")
        if args.hpp_mu is None:
            print("Warning: --hpp-file provided without --hpp-mu; HPP will not be scaled per BX.")
            hpp_scale = 1.0
        else:
            if num_hpp_events <= 0:
                print("Warning: HPP file had zero events; skipping HPP contribution.")
                hpp_counts = None
            else:
                hpp_scale = float(args.hpp_mu) / float(num_hpp_events)
                print(f"HPP normalization: mu={args.hpp_mu} events/BX, {num_hpp_events} events total, scale factor = {hpp_scale:.6g}")
        if hpp_counts is not None:
            for b in TRACKER_BRANCHES + CALO_BRANCHES:
                hpp_counts[b] = hpp_counts[b] * hpp_scale
            print("HPP normalization completed.")

    # Save plots and numpy arrays (titles show 1 bunch)
    for b, counts in hist_counts.items():
        plot_hist(
            b,
            counts,
            edges,
            args.outdir,
            plot_xmin=args.time_min,
            plot_xmax=get_tmax_for_branch(b),
            plot_binwidth=args.plot_binwidth,
            collider=args.collider,
            detector_version=args.detector,
            legend_fs=args.legend_fs,
            show_mean=args.show_mean,
            log_y=args.logy,
            counts_hpp=(hpp_counts[b] if (hpp_counts is not None and b in hpp_counts) else None),
            hpp_mu=args.hpp_mu,
        )
        tmax_used = get_tmax_for_branch(b)
        tmax_info = f" (tmax={tmax_used}ns)" if tmax_used != args.time_max else ""
        if tmax_used != args.time_max:
            print(f"Using custom time range for {b}: {args.time_min}-{tmax_used}ns (global: {args.time_min}-{args.time_max}ns)")
        print(f"Wrote plot for {b} to {args.outdir}/{b}.png and {args.outdir}/{b}.pdf{tmax_info}")

    # Save all histograms to a single NPZ for later inspection
    np.savez(
        args.out,
        edges=edges,
        files=np.array([p for _, p in (files if args.load_hist else files)], dtype=object),
        **{f"counts_{b}": v for b, v in hist_counts.items()},
        **({} if hpp_counts is None else {f"hpp_counts_{b}": v for b, v in hpp_counts.items()}),
        hpp_file=(args.hpp_file if args.hpp_file is not None else ''),
        hpp_mu=(args.hpp_mu if args.hpp_mu is not None else np.nan),
    )
    print(f"Wrote histograms to {args.out} and plots to {args.outdir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle

try:
    import mplhep as hep
    HAS_MPLHEP = True
except Exception:
    HAS_MPLHEP = False


def _load_ridge_json(path: str) -> Optional[Dict]:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        for key in ['ridge_theta', 'ridge_pt', 'line_theta', 'line_pt']:
            if key in data and data[key] is not None:
                data[key] = np.asarray(data[key], dtype=float)
            else:
                data[key] = None if key not in ('ridge_theta', 'ridge_pt') else np.asarray([])
        if 'power_law' in data and data['power_law'] is not None:
            A, B = data['power_law']
            data['power_law'] = (float(A), float(B))
        else:
            data['power_law'] = None
        data['metadata'] = data.get('metadata', {})
        return data
    except Exception as exc:
        print(f"Error reading ridge JSON '{path}': {exc}")
        return None


def _label_from_metadata(data: Dict, fallback: str) -> str:
    meta = data.get('metadata', {}) if data else {}
    collider = meta.get('collider')
    field_T = meta.get('field_T')
    label = fallback
    if collider and field_T is not None:
        #label = f"{collider} (B={field_T}T)"
        label = f"{collider}"
    elif collider:
        label = f"{collider}"
    return label


def _shade_blues(index: int, total: int):
    cmap = plt.get_cmap('Blues')
    vals = np.linspace(0.35, 0.9, max(total, 1))
    return cmap(vals[index % max(total, 1)])

def parse_args():
    p = argparse.ArgumentParser(description='Overlay deflection ridges from JSON and plot ROI and detector reach region in pT–theta (log–log).')
    p.add_argument('--ridge-250-PS1', required=True, help='Path to ridge JSON for C3 250 PS1')
    p.add_argument('--ridge-250-PS2', required=True, help='Path to ridge JSON for C3 250 PS2')
    p.add_argument('--ridge-550-PS1', required=True, help='Path to ridge JSON for C3 550 PS1')
    p.add_argument('--ridge-550-PS2', required=True, help='Path to ridge JSON for C3 550 PS2')

    # Optional PKL caches (to compute acceptance and overlay histograms)
    p.add_argument('--pkl-250-PS1', help='Path to PKL cache for C3 250 PS1')
    p.add_argument('--pkl-250-PS2', help='Path to PKL cache for C3 250 PS2')
    p.add_argument('--pkl-550-PS1', help='Path to PKL cache for C3 550 PS1')
    p.add_argument('--pkl-550-PS2', help='Path to PKL cache for C3 550 PS2')

    p.add_argument('--roi-theta-min', type=float, default=2e-3, help='Minimum theta (rad) for ROI')
    p.add_argument('--roi-theta-max', type=float, default=None, help='Maximum theta (rad) for ROI (omit for no upper bound)')
    p.add_argument('--roi-pt-min', type=float, default=1e-3, help='Minimum pT (GeV/c) for ROI')
    p.add_argument('--roi-pt-max', type=float, default=None, help='Maximum pT (GeV/c) for ROI (omit for no upper bound)')

    p.add_argument('--pt-det-min', type=float, default=None, help='Minimum pT (GeV/c) for vertex detector reach')
    p.add_argument('--theta-det-min', type=float, default=None, help='Minimum theta (rad) for vertex detector reach')

    p.add_argument('--theta-min', type=float, default=None, help='x-axis lower bound (theta)')
    p.add_argument('--theta-max', type=float, default=None, help='x-axis upper bound (theta)')
    p.add_argument('--pt-min', type=float, default=None, help='y-axis lower bound (pT)')
    p.add_argument('--pt-max', type=float, default=None, help='y-axis upper bound (pT)')
    
    # Optional: overlay one dataset's 2D histogram
    p.add_argument('--overlay-hist', choices=['250_PS1', '250_PS2', '550_PS1', '550_PS2'], help='Overlay 2D histogram (requires matching --pkl-...)')
    p.add_argument('--overlay-alpha', type=float, default=0.85, help='Alpha for overlay histogram')
    p.add_argument('--overlay-bins-theta', type=int, default=80, help='Theta bins for overlay histogram')
    p.add_argument('--overlay-bins-pt', type=int, default=80, help='pT bins for overlay histogram')
    p.add_argument('--overlay-cmap', default='viridis', help='Colormap for overlay histogram')
    p.add_argument('--overlay-colorbar', action='store_true', help='Show colorbar for overlay histogram')
    p.add_argument('--overlay-vmin', type=float, default=1e-1, help='LogNorm vmin for overlay histogram')
    p.add_argument('--overlay-vmax', type=float, default=None, help='LogNorm vmax for overlay histogram (omit for auto)')
    p.add_argument('--hide-overlay-hist', action='store_true', help='Suppress drawing overlay histograms')
    # Show fitted line overlay (optional)
    p.add_argument('--show-fit-line', action='store_true', help='If set, overlay the fitted ridge line (and annotation if available)')

    # Reachability boundary sampling options
    p.add_argument('--reachability-script', help='Path to reachability_analysis.py (defaults to sibling in pair_envelopes).')
    p.add_argument('--reachability-pt-min', type=float, help='Minimum pT to sample reachability boundary (GeV/c).')
    p.add_argument('--reachability-pt-max', type=float, help='Maximum pT to sample reachability boundary (GeV/c). Defaults to plot y-maximum.')
    p.add_argument('--reachability-pt-samples', type=int, default=400, help='Number of pT samples for reachability boundary.')
    p.add_argument('--reachability-linear-pt', action='store_true', help='Use linear spacing for reachability boundary sampling (default log).')
    p.add_argument('--reachability-charge', type=float, default=0.3, help='Effective charge factor q for reachability boundary (GeV*T*mm).')
    p.add_argument('--reachability-mag-field', type=float, default=5.0, help='Magnetic field B0 in Tesla for reachability boundary.')
    p.add_argument('--reachability-detector-radius', type=float, default=14.0, help='Detector radius in millimetres for reachability boundary.')
    p.add_argument('--reachability-z-max', type=float, default=76.0, help='Detector z-extent in millimetres for reachability boundary.')
    p.add_argument('--reachability-theta-upper', type=float, help='Optional theta upper bound (rad) when sampling reachability boundary.')

    p.add_argument('--out', required=True, help='Output plot file (e.g. PNG/PDF)')
    return p.parse_args()


def _valid_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x is None or y is None:
        return np.asarray([]), np.asarray([])
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[m], y[m]


def _draw_roi(ax, roi_theta: Tuple[Optional[float], Optional[float]], roi_pt: Tuple[Optional[float], Optional[float]]):
    th_min, th_max = roi_theta
    pt_min, pt_max = roi_pt
    scale = 1e3
    if th_min is not None and th_min > 0:
        ax.axvline(th_min, color='white', linestyle=':', linewidth=1, alpha=0.6)
    if th_max is not None and th_max > 0:
        ax.axvline(th_max, color='white', linestyle=':', linewidth=1, alpha=0.6)
    if pt_min is not None and pt_min > 0:
        ax.axhline(pt_min * scale, color='white', linestyle=':', linewidth=1, alpha=0.6)
    if pt_max is not None and pt_max > 0:
        ax.axhline(pt_max * scale, color='white', linestyle=':', linewidth=1, alpha=0.6)


def _resolve_reachability_path(custom_path: Optional[str]) -> Optional[str]:
    if custom_path:
        return custom_path
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(os.path.dirname(os.path.dirname(this_dir)), 'reachability_analysis.py')
    return candidate if os.path.isfile(candidate) else None


def _load_reachability_module(script_path: str):
    try:
        spec = importlib.util.spec_from_file_location('reachability_analysis_runtime', script_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        print(f"Warning: failed to load reachability module '{script_path}': {exc}")
        return None


def _sample_reachability_boundary(args, pt_upper_hint: float):
    script_path = _resolve_reachability_path(args.reachability_script)
    if not script_path:
        print('Warning: reachability_analysis.py not found; skipping detector reach boundary overlay.')
        return None, None
    print(f"Reachability script path: {script_path}")

    module = _load_reachability_module(script_path)
    if module is None:
        return None, None
    print("Reachability module loaded successfully.")

    compute_boundary = getattr(module, 'compute_boundary', None)
    if compute_boundary is None:
        print(f"Warning: reachability module '{script_path}' has no compute_boundary(); skipping overlay.")
        return None, None

    charge = args.reachability_charge
    mag_field = args.reachability_mag_field
    r_det = args.reachability_detector_radius
    z_max = args.reachability_z_max

    if charge <= 0 or mag_field <= 0 or r_det <= 0 or z_max <= 0:
        print('Warning: invalid reachability parameters; skipping boundary overlay.')
        return None, None

    pt_min_condition = (charge * mag_field * r_det) / 2000.0
    provided_pt_min = args.reachability_pt_min
    pt_min = max(provided_pt_min if provided_pt_min is not None else pt_min_condition, pt_min_condition)
    print(f"Reachability sampling pT_min condition: {pt_min_condition:.6e} GeV/c, using pT_min={pt_min:.6e} GeV/c")

    axis_pt_max = pt_upper_hint if (pt_upper_hint is not None and np.isfinite(pt_upper_hint)) else None
    pt_max = args.reachability_pt_max if args.reachability_pt_max is not None else axis_pt_max
    if pt_max is None or not np.isfinite(pt_max) or pt_max <= pt_min:
        pt_max = max(pt_min * 1.05, pt_min + 1e-3)
    print(f"Reachability sampling pT_max={pt_max:.6e} GeV/c (axis hint={axis_pt_max})")

    samples = max(int(args.reachability_pt_samples), 5)
    if args.reachability_linear_pt:
        pt_values = np.linspace(pt_min, pt_max, samples)
    else:
        pt_values = np.logspace(np.log10(pt_min), np.log10(pt_max), samples)
    if pt_values.size > 0:
        pt_values[0] = pt_min
        pt_values[-1] = pt_max
    print(f"Reachability sampling {samples} points (log spacing={not args.reachability_linear_pt})")

    try:
        theta_values = compute_boundary(
            pt_values,
            q=charge,
            B0=mag_field,
            r_det=r_det,
            z_max=z_max,
            theta_upper=args.reachability_theta_upper,
        )
    except Exception as exc:
        print(f"Warning: failed to compute reachability boundary: {exc}")
        return None, None

    mask = (
        np.isfinite(theta_values)
        & np.isfinite(pt_values)
        & (theta_values > 0)
        & (pt_values > 0)
    )
    if not np.any(mask):
        print('Warning: no valid reachability boundary points computed; skipping overlay.')
        return None, None

    theta_valid = theta_values[mask]
    pt_valid = pt_values[mask]
    order = np.argsort(theta_valid)
    theta_sorted = theta_valid[order]
    pt_sorted = pt_valid[order]

    print(
        "Reachability samples: {} points, theta range [{:.3e}, {:.3e}], pT range [{:.3e}, {:.3e}] GeV/c".format(
            theta_sorted.size,
            float(theta_sorted.min()),
            float(theta_sorted.max()),
            float(pt_sorted.min()),
            float(pt_sorted.max()),
        )
    )

    return theta_sorted, pt_sorted


def main():
    args = parse_args()

    if HAS_MPLHEP:
        hep.style.use("CMS")

    j250_ps1 = _load_ridge_json(args.ridge_250_PS1)
    j250_ps2 = _load_ridge_json(args.ridge_250_PS2)
    j550_ps1 = _load_ridge_json(args.ridge_550_PS1)
    j550_ps2 = _load_ridge_json(args.ridge_550_PS2)

    # Load pT/theta samples from PKL caches (if provided)
    def _load_pt_theta_from_pkl(pkl_path: Optional[str]):
        if not pkl_path:
            return np.asarray([]), np.asarray([])
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'combined_hist' in data:
                comb = data['combined_hist']
                pt_vals = np.asarray(comb.get('pt_values', []), dtype=float)
                th_vals = np.asarray(comb.get('theta_values', []), dtype=float)
                return pt_vals, th_vals
            if isinstance(data, dict):
                pt_vals = np.asarray(data.get('pt_values', []), dtype=float)
                th_vals = np.asarray(data.get('theta_values', []), dtype=float)
                return pt_vals, th_vals
        except Exception as exc:
            print(f"Warning: failed to read PKL '{pkl_path}': {exc}")
        return np.asarray([]), np.asarray([])

    pkl_pt_th = {
        '250_PS1': _load_pt_theta_from_pkl(args.pkl_250_PS1),
        '250_PS2': _load_pt_theta_from_pkl(args.pkl_250_PS2),
        '550_PS1': _load_pt_theta_from_pkl(args.pkl_550_PS1),
        '550_PS2': _load_pt_theta_from_pkl(args.pkl_550_PS2),
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\theta$ [rad]', fontsize=17)
    ax.set_ylabel(r'$p_{\mathrm{T}}$ [MeV/c]', fontsize=17)
    ax.grid(True, which='both', alpha=0.3)

    label_order = ['250_PS1', '250_PS2', '550_PS1', '550_PS2']
    colors = {lab: _shade_blues(i, len(label_order)) for i, lab in enumerate(label_order)}

    legend_handles = []
    legend_labels = []

    def plot_one(data, fallback_key, color, linestyle='-'):
        if data is None:
            return
        th, pt = _valid_xy(data.get('ridge_theta'), data.get('ridge_pt'))
        if th.size == 0:
            return
        pt_mev = pt * 1e3
        line, = ax.plot(th, pt_mev, color=color, linewidth=2, linestyle=linestyle)
        legend_handles.append(line)
        legend_labels.append(_label_from_metadata(data, fallback_key))

        if args.show_fit_line:
            line_th, line_pt = _valid_xy(data.get('line_theta'), data.get('line_pt'))
            if line_th.size > 0:
                fit_line, = ax.plot(line_th, line_pt * 1e3, color=color, linestyle='--', linewidth=1.5)
                legend_handles.append(fit_line)
                if data.get('power_law') is not None:
                    A, B = data['power_law']
                    A_mev = A * 1e3
                    ax.text(
                        0.03, 0.04,
                        fr'$p_{{\mathrm{T}}} \approx {A_mev:.2g}\,\theta^{{{B:.2f}}}\ \mathrm{{MeV}}/c$',
                        transform=ax.transAxes,
                        fontsize=14,
                        color=color,
                        bbox=dict(facecolor='white', alpha=0.12, edgecolor='none')
                    )

    plot_one(j250_ps1, 'C3 250 PS1', colors['250_PS1'], linestyle='-')
    plot_one(j250_ps2, 'C3 250 PS2', colors['250_PS2'], linestyle='--')
    plot_one(j550_ps1, 'C3 550 PS1', colors['550_PS1'], linestyle='-')
    plot_one(j550_ps2, 'C3 550 PS2', colors['550_PS2'], linestyle='--')

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Enforce ROI-only axis range when ROI values are provided
    if args.roi_theta_min is not None and args.roi_theta_min > 0:
        xmin = args.roi_theta_min
    if args.roi_theta_max is not None and args.roi_theta_max > 0:
        xmax = args.roi_theta_max
    if args.roi_pt_min is not None and args.roi_pt_min > 0:
        ymin = args.roi_pt_min * 1e3
    if args.roi_pt_max is not None and args.roi_pt_max > 0:
        ymax = args.roi_pt_max * 1e3

    # Allow explicit overrides via direct axis options if user set them
    if args.theta_min is not None and args.theta_min > 0:
        xmin = args.theta_min
    if args.theta_max is not None and args.theta_max > 0:
        xmax = args.theta_max
    if args.pt_min is not None and args.pt_min > 0:
        ymin = args.pt_min * 1e3
    if args.pt_max is not None and args.pt_max > 0:
        ymax = args.pt_max * 1e3

    if xmax <= xmin:
        xmax = xmin * 10.0
    if ymax <= ymin:
        ymax = ymin * 10.0
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Optional overlay histogram for a selected dataset
    def _overlay_hist_for(key: str):
        pt, th = pkl_pt_th.get(key, (np.asarray([]), np.asarray([])))
        if pt.size == 0 or th.size == 0:
            print(f"Note: overlay requested for {key} but no PKL data available; skipping overlay.")
            return None
        m = np.isfinite(pt) & np.isfinite(th) & (pt > 0) & (th > 0)
        if not np.any(m):
            print(f"Note: overlay requested for {key} but no finite positive entries; skipping overlay.")
            return None
        pt_ = pt[m]
        th_ = th[m]
        pt_plot = pt_ * 1e3
        th_min = max(th_.min(), 1e-9)
        th_max = th_.max()
        pt_min = max(pt_plot.min(), 1e-6)
        pt_max = pt_plot.max()
        th_bins = np.logspace(np.log10(th_min), np.log10(th_max), max(5, int(args.overlay_bins_theta)))
        pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), max(5, int(args.overlay_bins_pt)))
        hist = ax.hist2d(
            th_, pt_plot, bins=(th_bins, pt_bins),
            norm=LogNorm(vmin=args.overlay_vmin, vmax=args.overlay_vmax),
            cmap=args.overlay_cmap, alpha=args.overlay_alpha, zorder=0
        )
        if args.overlay_colorbar:
            cbar = fig.colorbar(hist[3], ax=ax)
            cbar.set_label('Entries/bin', fontsize=17)
        return hist

    if args.overlay_hist is not None and not args.hide_overlay_hist:
        _overlay_hist_for(args.overlay_hist)
        # Re-apply ROI axis limits because hist2d can autoscale the axes
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    _draw_roi(ax, (args.roi_theta_min, args.roi_theta_max), (args.roi_pt_min, args.roi_pt_max))

    reach_theta, reach_pt = _sample_reachability_boundary(args, ax.get_ylim()[1])
    if reach_theta is not None and reach_pt is not None:
        x_top = ax.get_xlim()[1]
        if x_top > reach_theta[-1]:
            print(f"Extending reachability boundary to theta={x_top:.6e} rad with pT={reach_pt[-1]:.6e} GeV/c")
            reach_theta = np.append(reach_theta, x_top)
            reach_pt = np.append(reach_pt, reach_pt[-1])

        y_top = ax.get_ylim()[1]
        reach_pt_mev = reach_pt * 1e3
        reach_line, = ax.plot(
            reach_theta,
            reach_pt_mev,
            color='black',
            linewidth=2.3,
            linestyle='-',
            alpha=0.32,
            zorder=3,
        )
        ax.fill_between(
            reach_theta,
            reach_pt_mev,
            np.full_like(reach_pt_mev, y_top),
            color='black',
            alpha=0.05,
            edgecolor='none',
            zorder=2,
        )
        legend_handles.append(reach_line)
        legend_labels.append('vertex reach boundary')

    # Compute acceptance fractions if PKL data available
    def _compute_acceptance_fraction(
        pt: np.ndarray,
        th: np.ndarray,
        boundary_theta: Optional[np.ndarray],
        boundary_pt: Optional[np.ndarray],
        pt_min_cut: Optional[float],
        th_min_cut: Optional[float],
    ) -> Optional[float]:
        if pt.size == 0 or th.size == 0:
            return None
        m = np.isfinite(pt) & np.isfinite(th) & (pt > 0) & (th > 0)
        if not np.any(m):
            return None

        pt_valid = pt[m]
        th_valid = th[m]
        total = pt_valid.size

        if (
            boundary_theta is not None
            and boundary_pt is not None
            and boundary_theta.size > 1
            and boundary_pt.size > 1
        ):
            pt_threshold = np.interp(th_valid, boundary_theta, boundary_pt, left=boundary_pt[0], right=boundary_pt[-1])
            below_range = th_valid < boundary_theta[0]
            pt_threshold[below_range] = np.nan
            acc_mask = np.isfinite(pt_threshold) & (pt_valid >= pt_threshold)
            num = int(np.count_nonzero(acc_mask))
            return num / total if total > 0 else None

        acc_mask = np.ones_like(pt_valid, dtype=bool)
        if pt_min_cut is not None and pt_min_cut > 0:
            acc_mask &= pt_valid >= pt_min_cut
        if th_min_cut is not None and th_min_cut > 0:
            acc_mask &= th_valid >= th_min_cut
        num = int(np.count_nonzero(acc_mask))
        return num / total if total > 0 else None

    fractions = {}
    for key in ['250_PS1', '250_PS2', '550_PS1', '550_PS2']:
        pt, th = pkl_pt_th.get(key, (np.asarray([]), np.asarray([])))
        frac = _compute_acceptance_fraction(
            pt,
            th,
            reach_theta,
            reach_pt,
            args.pt_det_min,
            args.theta_det_min,
        )
        fractions[key] = frac
        if frac is not None:
            print(f"Acceptance fraction for {key}: {frac:.4f} ({frac*100:.2f}%)")
        else:
            print(f"Acceptance fraction for {key}: N/A (missing data)")

    # Append fractions to legend labels if we can infer mapping
    remapped_labels = []
    for lbl in legend_labels:
        lbl_new = lbl
        if '250' in lbl and 'PS1' in lbl:
            f = fractions.get('250_PS1')
            if f is not None:
                lbl_new = f"{lbl}  (fraction: {f*100:.2f}%)"
        elif '250' in lbl and 'PS2' in lbl:
            f = fractions.get('250_PS2')
            if f is not None:
                lbl_new = f"{lbl}  (fraction: {f*100:.2f}%)"
        elif '550' in lbl and 'PS1' in lbl:
            f = fractions.get('550_PS1')
            if f is not None:
                lbl_new = f"{lbl}  (fraction: {f*100:.2f}%)"
        elif '550' in lbl and 'PS2' in lbl:
            f = fractions.get('550_PS2')
            if f is not None:
                lbl_new = f"{lbl}  (fraction: {f*100:.2f}%)"
        remapped_labels.append(lbl_new)
    legend_labels = remapped_labels

    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc='upper left', fontsize=15, frameon=True, fancybox=True, shadow=True)

    fig.tight_layout()
    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved ridge overlay plot to {out_path}")


if __name__ == '__main__':
    main()

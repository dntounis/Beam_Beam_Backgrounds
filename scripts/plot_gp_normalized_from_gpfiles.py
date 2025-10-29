#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Optional, Tuple

import colorsys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

try:
    import mplhep as hep
    HAVE_MPLHEP = True
    hep.style.use(hep.style.ATLAS)
except Exception:
    HAVE_MPLHEP = False


def parse_set(arg: str) -> Dict[str, str]:
    """
    Parse --set of the form:
      LABEL:ROOTS:dir=/path/to/compare_output[,prefix=OUT_PREFIX]

    Returns dict with keys: label, roots, dir, prefix(optional).
    """
    try:
        label, roots_str, rest = arg.split(":", 2)
        kv = {}
        for item in rest.split(","):
            if not item:
                continue
            k, v = item.split("=", 1)
            kv[k.strip()] = v.strip()
        if "dir" not in kv:
            raise ValueError("missing dir=")
        return {
            "label": label.strip(),
            "roots": roots_str.strip(),
            "dir": kv["dir"],
            "prefix": kv.get("prefix", "").strip(),
        }
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "Invalid --set. Use LABEL:ROOTS:dir=PATH[,prefix=OUT_PREFIX]"
        ) from exc


def _format_set_label(label: str) -> str:
    parts = label.split("_")
    if len(parts) >= 2:
        txt = f"{parts[0]}-{parts[1]}"
        if len(parts) > 2:
            txt += " " + " ".join(parts[2:])
        return txt
    return label


def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isfile(p) and os.path.getsize(p) > 0:
            return p
    return None


def find_gp_hist(dir_path: str, prefix: str, species: str, axis: str, stem: str = "") -> Optional[str]:
    """
    Find a GP histogram file produced by compare step.
    axis in {"x","y"}.
    Tries common naming patterns:
      - {prefix}.{axis}{_stem}.{species}.gp
      - {axis}{_stem}.{species}.gp
      - {axis}_fine.{species}.gp (fallback)
    """
    candidates = []
    suf = f"_{stem}" if stem else ""
    if prefix:
        candidates.append(os.path.join(dir_path, f"{prefix}.{axis}{suf}.{species}.gp"))
    candidates.extend([
        os.path.join(dir_path, f"{axis}{suf}.{species}.gp"),
        os.path.join(dir_path, f"{axis}_fine.{species}.gp"),
    ])
    return _first_existing(candidates)


def load_hist_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        data = np.loadtxt(path)
        if data.size == 0:
            return np.array([]), np.array([])
        if data.ndim == 1:
            data = data.reshape(-1, 2)
        # Keep first two columns (x, y)
        if data.shape[1] >= 2:
            x, y = data[:, 0], data[:, 1]
        else:
            return np.array([]), np.array([])
        # Filter finite
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]
    except Exception:
        return np.array([]), np.array([])


def normalize_curve(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0:
        return x, y
    # Ensure sorted by x for integration
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    area = float(np.trapezoid(ys, xs))
    if area > 0 and np.isfinite(area):
        ys = ys / area
    return xs, ys

def resample_curve(xs: np.ndarray, ys: np.ndarray, nbins: int) -> Tuple[np.ndarray, np.ndarray]:
    if xs.size == 0 or ys.size == 0 or nbins is None or nbins <= 2:
        return xs, ys
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    xnew = np.linspace(xmin, xmax, nbins)
    ynew = np.interp(xnew, xs, ys)
    # Re-normalize to keep unit area
    area = float(np.trapezoid(ynew, xnew))
    if area > 0 and np.isfinite(area):
        ynew = ynew / area
    return xnew, ynew

def find_events_file(dir_path: str, prefix: str, species: str) -> Optional[str]:
    """Locate the GP events file written by compare script (e.g., ee.events.gp)."""
    candidates = []
    if prefix:
        # Less common, but try a prefixed variant
        candidates.append(os.path.join(dir_path, f"{prefix}.{species}.events.gp"))
    candidates.append(os.path.join(dir_path, f"{species}.events.gp"))
    return _first_existing(candidates)

def load_events_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        data = np.loadtxt(path)
        if data.size == 0:
            return np.array([]), np.array([])
        if data.ndim == 1:
            data = data.reshape(-1, max(2, data.shape[0]))
        # Use first two columns as x,y (third column is weight=1)
        if data.shape[1] >= 2:
            x, y = data[:, 0], data[:, 1]
        else:
            return np.array([]), np.array([])
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]
    except Exception:
        return np.array([]), np.array([])


def main():
    ap = argparse.ArgumentParser(description="Overlay normalized GP x,y spectra from .gp histogram files.")
    ap.add_argument(
        "--set",
        action="append",
        type=parse_set,
        required=True,
        help="LABEL:ROOTS:dir=PATH[,prefix=OUT_PREFIX] (ROOTS used only for label)",
    )
    ap.add_argument("--out", default="gp_norm_from_gp", help="Output basename (no extension)")
    ap.add_argument("--format", default="pdf", choices=["pdf","png","svg"], help="Output format")
    ap.add_argument("--logy", action="store_true", help="Use logarithmic scale on Y axis")
    ap.add_argument("--ymin", type=float, default=1e-4, help="Minimum Y value for axis (default: 1e-4)")
    # Optional per-species target bin counts (resampling). If unset, keep native binning from .gp files.
    ap.add_argument("--bins-ee", type=int, default=None, help="Target bins for ee (optional)")
    ap.add_argument("--bins-eg", type=int, default=None, help="Target bins for eg (optional)")
    ap.add_argument("--bins-ge", type=int, default=None, help="Target bins for ge (optional)")
    ap.add_argument("--bins-gg", type=int, default=None, help="Target bins for gg (optional)")
    args = ap.parse_args()

    def adjust_lightness(color_hex: str, delta: float) -> str:
        """Shift perceived lightness while keeping hue for accessibility."""
        r, g, b = mcolors.to_rgb(color_hex)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l = max(0.0, min(1.0, l + delta))
        return mcolors.to_hex(colorsys.hls_to_rgb(h, l, s))

    def color_for_species(species_key: str, idx: int, total: int) -> str:
        base_colors = {
            "ee": "#0072B2",     # blue (Okabe-Ito)
            "eg": "#E69F00",     # orange
            "ge": "#009E73",     # bluish green
            "gg": "#D55E00",     # vermillion
            "eg_ge": "#009E73",  # shared for merged eg/ge
        }
        base = base_colors.get(species_key, "#595959")
        if total <= 1:
            return base
        idx = idx % total
        delta_values = np.linspace(-0.18, 0.28, total)
        return adjust_lightness(base, float(delta_values[idx]))

    def linestyle_for_label(label: str) -> str:
        style_map = {
            "C3_250_PS1": "-",
            "C3_250_PS2": "--",
            "C3_550_PS1": ":",
            "C3_550_PS2": "-.",
        }
        return style_map.get(label, "-")

    species_order = [
        ("ee", "ee"),
        ("eg", "eγ"),
        ("ge", "γe"),
        ("gg", "γγ"),
    ]

    num_sets = len(args.set)

    # Two figures: x and y
    figx, axx = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    figy, axy = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    # One more figure: normalized sqrt(s') where s' = s * x * y, so sqrt(s')/sqrt(s) = sqrt(x*y)
    figs, axs = plt.subplots(figsize=(7.0, 4.6), dpi=150)

    # Map species to requested bins
    species_bins = {
        "ee": args.bins_ee,
        "eg": args.bins_eg,
        "ge": args.bins_ge,
        "gg": args.bins_gg,
    }

    # Reference indices to keep styling consistent for known parameter sets
    set_order_index = {
        "C3_250_PS1": 0,
        "C3_250_PS2": 2,
        "C3_550_PS1": 1,
        "C3_550_PS2": 3,
    }
    for idx, s in enumerate(args.set):
        label = s["label"]
        label_disp = _format_set_label(label)
        ls = linestyle_for_label(label)
        d = s["dir"]
        pfx = s.get("prefix", "")

        for sp_key, sp_disp in species_order:
            color = color_for_species(sp_key, idx, num_sets)
            # x-spectra
            xfile = find_gp_hist(d, pfx, sp_key, axis="x")
            if xfile:
                x, y = load_hist_xy(xfile)
                xs, ys = normalize_curve(x, y)
                nb = species_bins.get(sp_key)
                if nb is not None:
                    xs, ys = resample_curve(xs, ys, nb)
                if xs.size:
                    axx.plot(xs, ys, color=color, ls=ls, lw=1.3, label=f"{sp_disp} ({label_disp})")
            # y-spectra
            yfile = find_gp_hist(d, pfx, sp_key, axis="y")
            if yfile:
                x2, y2 = load_hist_xy(yfile)
                xs2, ys2 = normalize_curve(x2, y2)
                nb = species_bins.get(sp_key)
                if nb is not None:
                    xs2, ys2 = resample_curve(xs2, ys2, nb)
                if xs2.size:
                    axy.plot(xs2, ys2, color=color, ls=ls, lw=1.3, label=f"{sp_disp} ({label_disp})")

        # s' (center-of-mass fraction) from event clouds: build once per set per species-group
        # 1) ee
        ee_ev = find_events_file(d, pfx, "ee")
        if ee_ev:
            ex, ey = load_events_xy(ee_ev)
            if ex.size and ey.size:
                sfrac = np.sqrt(np.clip(ex * ey, 0.0, None))
                nb = species_bins.get("ee") or 200
                hist, edges = np.histogram(sfrac, bins=nb, range=(0.0, 1.0), density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
                set_idx = set_order_index.get(label, idx)
                color_ee = color_for_species("ee", set_idx, num_sets)
                axs.plot(centers, hist, color=color_ee, ls=ls, lw=1.3, label=f"ee ({label_disp})")
        # 2) eg + ge merged
        eg_ev = find_events_file(d, pfx, "eg")
        ge_ev = find_events_file(d, pfx, "ge")
        s_list = []
        for evp in (eg_ev, ge_ev):
            if evp:
                ex, ey = load_events_xy(evp)
                if ex.size and ey.size:
                    s_list.append(np.sqrt(np.clip(ex * ey, 0.0, None)))
        if s_list:
            sfrac = np.concatenate(s_list)
            nb = species_bins.get("eg") or 200
            hist, edges = np.histogram(sfrac, bins=nb, range=(0.0, 1.0), density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            set_idx = set_order_index.get(label, idx)
            color_eg = color_for_species("eg_ge", set_idx, num_sets)
            axs.plot(centers, hist, color=color_eg, ls=ls, lw=1.3, label=f"eγ/γe ({label_disp})")
        # 3) gg
        gg_ev = find_events_file(d, pfx, "gg")
        if gg_ev:
            ex, ey = load_events_xy(gg_ev)
            if ex.size and ey.size:
                sfrac = np.sqrt(np.clip(ex * ey, 0.0, None))
                nb = species_bins.get("gg") or 200
                hist, edges = np.histogram(sfrac, bins=nb, range=(0.0, 1.0), density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
                set_idx = set_order_index.get(label, idx)
                color_gg = color_for_species("gg", set_idx, num_sets)
                axs.plot(centers, hist, color=color_gg, ls=ls, lw=1.3, label=f"γγ ({label_disp})")

    # Style and legends
    for ax, axis_label in ((axx, "x"), (axy, "y")):
        ax.set_xlabel(axis_label)
        ax.set_ylabel("Normalized counts")
        if args.logy:
            ax.set_yscale("log")
        if args.ymin is not None:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(bottom=args.ymin, top=ymax)
        ax.grid(alpha=0.25)
        # Deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uh, ul = [], []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            uh.append(h)
            ul.append(l)
        ax.legend(uh, ul, ncol=2, fontsize=8, frameon=True)

    # Style s' figure
    axs.set_xlabel(r"$\sqrt{s'}/\sqrt{s} = \sqrt{x\,y}$")
    axs.set_ylabel("Normalized counts")
    if args.logy:
        axs.set_yscale("log")
    if args.ymin is not None:
        ymin, ymax = axs.get_ylim()
        axs.set_ylim(bottom=args.ymin, top=ymax)
    axs.grid(alpha=0.25)
    # Build three compact sub-legends (ee, eγ/γe, γγ)
    ncol_sets = max(1, min(4, len(args.set)))
    # Collect all lines for each group by re-parsing current handles/labels and matching colors to palettes is brittle.
    # Instead, rebuild grouped legends by capturing handles at plotting time. For backward compatibility with above, reuse current handles when needed.
    # Create grouped legends positioned outside on the right.
    # Gather handles/labels by color groups based on label prefixes we used when plotting.
    # We already have distinct labels like "ee (...)" etc. Split by prefix safely.
    h_all, l_all = axs.get_legend_handles_labels()
    group_to_handles = {"ee": [], "eγ/γe": [], "γγ": []}
    group_to_labels = {"ee": [], "eγ/γe": [], "γγ": []}
    for h, lab in zip(h_all, l_all):
        if lab.startswith("ee "):
            group_to_handles["ee"].append(h)
            group_to_labels["ee"].append(lab.replace("ee ", "").strip("()"))
        elif lab.startswith("eγ/γe "):
            group_to_handles["eγ/γe"].append(h)
            group_to_labels["eγ/γe"].append(lab.replace("eγ/γe ", "").strip("()"))
        elif lab.startswith("γγ "):
            group_to_handles["γγ"].append(h)
            group_to_labels["γγ"].append(lab.replace("γγ ", "").strip("()"))

    if group_to_handles["ee"]:
        leg_ee = axs.legend(group_to_handles["ee"], group_to_labels["ee"], title="ee", ncol=1,
                            fontsize=8, title_fontsize=9, frameon=True,
                            loc='upper left', bbox_to_anchor=(0.10, 0.98), borderaxespad=0.)
        axs.add_artist(leg_ee)
    if group_to_handles["eγ/γe"]:
        leg_eg = axs.legend(group_to_handles["eγ/γe"], group_to_labels["eγ/γe"], title="eγ/γe", ncol=1,
                            fontsize=8, title_fontsize=9, frameon=True,
                            loc='upper left', bbox_to_anchor=(0.40, 0.98), borderaxespad=0.)
        axs.add_artist(leg_eg)
    if group_to_handles["γγ"]:
        leg_gg = axs.legend(group_to_handles["γγ"], group_to_labels["γγ"], title="γγ", ncol=1,
                            fontsize=8, title_fontsize=9, frameon=True,
                            loc='upper left', bbox_to_anchor=(0.70, 0.98), borderaxespad=0.)
        axs.add_artist(leg_gg)

    outx = f"{args.out}.x.{args.format}"
    outy = f"{args.out}.y.{args.format}"
    outs = f"{args.out}.sfrac.{args.format}"
    figx.tight_layout(); figx.savefig(outx)
    figy.tight_layout(); figy.savefig(outy)
    figs.tight_layout(); figs.savefig(outs)
    plt.close(figx); plt.close(figy); plt.close(figs)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import mplhep as hep
    HAVE_MPLHEP = True
    hep.style.use(hep.style.ATLAS)
except Exception:
    HAVE_MPLHEP = False


Species = Tuple[str, str]  # (key, display)


def parse_set_arg(arg: str) -> Dict[str, str]:
    """
    Parse one --set argument of the form:
      LABEL:ROOTS:ee=/path/ee.out,eg=/path/eg.out,ge=/path/ge.out,gg=/path/gg.out

    Returns a dict with keys: label, roots, ee, eg, ge, gg
    """
    try:
        label, roots_str, rest = arg.split(":", 2)
        kv = {}
        for item in rest.split(","):
            k, v = item.split("=", 1)
            kv[k.strip()] = v.strip()
        result = {
            "label": label.strip(),
            "roots": roots_str.strip(),
            "ee": kv.get("ee", ""),
            "eg": kv.get("eg", ""),
            "ge": kv.get("ge", ""),
            "gg": kv.get("gg", ""),
        }
        # Basic validation
        if not result["label"] or not result["roots"]:
            raise ValueError
        return result
    except Exception:
        raise argparse.ArgumentTypeError(
            "Invalid --set format. Expected LABEL:ROOTS:ee=...,eg=...,ge=...,gg=..."
        )


def _format_set_label(label: str) -> str:
    parts = label.split("_")
    if len(parts) >= 2:
        txt = f"{parts[0]}-{parts[1]}"
        if len(parts) > 2:
            txt += " " + " ".join(parts[2:])
        return txt
    return label


def _read_two_column_file(path: str) -> np.ndarray:
    """Load a two-column whitespace file robustly, return Nx2 float array or empty array."""
    try:
        data = np.loadtxt(path)
        if data.size == 0:
            return np.empty((0, 2), dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 2)
        # Keep only first two columns
        if data.shape[1] > 2:
            data = data[:, :2]
        return data
    except Exception:
        return np.empty((0, 2), dtype=float)


def compute_normalized_hist(values: np.ndarray, bins: int, rng: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(values, bins=bins, range=rng, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def build_spectra(files_by_species: Dict[str, str], roots: float, bins: int, frac_range: Tuple[float, float]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    For a set, build x and y normalized spectra per species.
    Returns dict species -> (x_centers, x_counts, y_centers, y_counts)
    """
    spectra = {}
    scale = roots / 2.0
    for sp, path in files_by_species.items():
        if not path:
            continue
        data = _read_two_column_file(path)
        if data.size == 0:
            spectra[sp] = (np.array([]), np.array([]), np.array([]), np.array([]))
            continue
        # Raw energies in GeV → fractional energies x,y in [0, ~1]
        xvals = 2.0 * data[:, 0] / roots
        yvals = 2.0 * data[:, 1] / roots
        # Filter to finite and within a padded range to avoid outliers
        maskx = np.isfinite(xvals)
        masky = np.isfinite(yvals)
        xvals = xvals[maskx]
        yvals = yvals[masky]
        x_centers, x_counts = compute_normalized_hist(xvals, bins=bins, rng=frac_range)
        y_centers, y_counts = compute_normalized_hist(yvals, bins=bins, rng=frac_range)
        spectra[sp] = (x_centers, x_counts, y_centers, y_counts)
    return spectra


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Plot normalized GuineaPig lumi spectra (x and y) overlaid in one figure. "
            "Provide multiple --set entries; colors map species (ee,eg,ge,gg), line styles map parameter sets."
        )
    )
    ap.add_argument(
        "--set",
        action="append",
        type=parse_set_arg,
        required=True,
        help=(
            "Parameter set spec: LABEL:ROOTS:ee=PATH,eg=PATH,ge=PATH,gg=PATH. "
            "Example: C3_250_PS1:250:ee=.../lumi.ee.out,eg=.../lumi.eg.out,ge=.../lumi.ge.out,gg=.../lumi.gg.out"
        ),
    )
    ap.add_argument("--bins", type=int, default=200, help="Number of bins for spectra (default: 200)")
    ap.add_argument(
        "--range",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        metavar=("LOW", "HIGH"),
        help="Range in fractional energy for histograms (default: 0 1)",
    )
    ap.add_argument("--out", default="gp_normalized_spectra", help="Output basename (no extension)")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"], help="Output format")
    args = ap.parse_args()

    species_list: List[Species] = [
        ("ee", "ee"),
        ("eg", "eγ"),
        ("ge", "γe"),
        ("gg", "γγ"),
    ]
    color_by_species = {
        "ee": "tab:blue",
        "eg": "tab:orange",
        "ge": "tab:green",
        "gg": "tab:red",
    }
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    # Prepare figures: one for x, one for y
    fig_x, ax_x = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    fig_y, ax_y = plt.subplots(figsize=(7.0, 4.6), dpi=150)

    # Iterate sets
    for idx, s in enumerate(args.set):
        label = s["label"]
        label_disp = _format_set_label(label)
        try:
            roots = float(s["roots"])
        except Exception:
            raise SystemExit(f"Invalid ROOTS for set {label}: {s['roots']}")
        files = {sp: s.get(sp, "") for sp, _ in species_list}

        spectra = build_spectra(files, roots=roots, bins=args.bins, frac_range=tuple(args.range))
        ls = linestyles[idx % len(linestyles)]

        for sp_key, sp_disp in species_list:
            col = color_by_species[sp_key]
            x_c, x_h, y_c, y_h = spectra.get(sp_key, (np.array([]),)*4)
            if x_c.size > 0 and x_h.size > 0:
                ax_x.plot(x_c, x_h, color=col, ls=ls, lw=1.3, label=f"{sp_disp} ({label_disp})")
            if y_c.size > 0 and y_h.size > 0:
                ax_y.plot(y_c, y_h, color=col, ls=ls, lw=1.3, label=f"{sp_disp} ({label_disp})")

    # Styling
    for ax, axis_label in ((ax_x, "x"), (ax_y, "y")):
        ax.set_xlabel(axis_label)
        ax.set_ylabel("1/N dN/d" + axis_label)
        ax.set_xlim(args.range[0], args.range[1])
        ax.grid(alpha=0.25)
        # Single merged legend
        handles, labels = ax.get_legend_handles_labels()
        # Deduplicate legend entries while preserving order
        seen = set()
        uniq_handles = []
        uniq_labels = []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            uniq_handles.append(h)
            uniq_labels.append(l)
        ax.legend(uniq_handles, uniq_labels, ncol=2, fontsize=8, frameon=True)

    # Save
    out_x = f"{args.out}.x.{args.format}"
    out_y = f"{args.out}.y.{args.format}"
    fig_x.tight_layout()
    fig_y.tight_layout()
    fig_x.savefig(out_x)
    fig_y.savefig(out_y)
    plt.close(fig_x)
    plt.close(fig_y)


if __name__ == "__main__":
    main()



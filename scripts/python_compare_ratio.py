#!/usr/bin/env python3
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import mplhep as hep
    HAVE_MPLHEP = True
    # Apply ATLAS style globally
    hep.style.use(hep.style.ATLAS)
except Exception:
    HAVE_MPLHEP = False

def load_1d(path):
    try:
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data.reshape(-1, 2)
        if data.shape[1] < 2:
            return None, None
        return data[:, 0], data[:, 1]
    except Exception:
        return None, None

def load_2d(path):
    try:
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data.reshape(-1, 3)
        if data.shape[1] < 3:
            return None, None, None
        xs = np.unique(data[:, 0])
        ys = np.unique(data[:, 1])
        Z = np.full((ys.size, xs.size), np.nan)
        xi = {v: i for i, v in enumerate(xs)}
        yi = {v: i for i, v in enumerate(ys)}
        for x, y, z in data:
            Z[yi[y], xi[x]] = z
        return xs, ys, Z
    except Exception:
        return None, None, None

def align_to_gp(x_gp, y_gp, x_c2, y_c2):
    if x_gp is None or y_gp is None:
        return None, None, None
    if x_c2 is None or y_c2 is None:
        return x_gp, y_gp, None
    # If grids match, return directly; else interpolate Circe2 onto GP x-grid
    if len(x_gp) == len(x_c2) and np.allclose(x_gp, x_c2, rtol=0, atol=1e-12):
        y_c2_on_gp = y_c2
    else:
        # Require monotonic x for interpolation
        order_gp = np.argsort(x_gp)
        order_c2 = np.argsort(x_c2)
        x_gp_s, y_gp_s = x_gp[order_gp], y_gp[order_gp]
        x_c2_s, y_c2_s = x_c2[order_c2], y_c2[order_c2]
        # Clip interp range to Circe2 support to avoid NaNs at edges
        xmin, xmax = x_c2_s[0], x_c2_s[-1]
        xq = np.clip(x_gp_s, xmin, xmax)
        y_c2_on_gp_s = np.interp(xq, x_c2_s, y_c2_s, left=np.nan, right=np.nan)
        # Undo sort to original GP order
        inv = np.empty_like(order_gp)
        inv[order_gp] = np.arange(order_gp.size)
        y_c2_on_gp = y_c2_on_gp_s[inv]
        x_gp_s = x_gp_s  # unused beyond this point
    return x_gp, y_gp, y_c2_on_gp

def _format_label_from_prefix(prefix: str) -> str:
    if not prefix:
        return ""
    parts = prefix.split("_")
    if len(parts) >= 2:
        label = f"{parts[0]}-{parts[1]}"
        if len(parts) > 2:
            label += " " + " ".join(parts[2:])
        return label
    return prefix

def _draw_collider_label(ax, collider_label: str):
    if not collider_label:
        return
    # Place in the top-left of the axes
    ax.text(0.02, 0.98, collider_label, transform=ax.transAxes,
            ha="left", va="top")

def plot_1d_with_ratio(gp_path, c2_path, title, out_path, logy=True, ratio_ylim=None, collider_label: str = ""):
    xg, yg = load_1d(gp_path)
    xc, yc = load_1d(c2_path)
    xg, yg, yc_on_gp = align_to_gp(xg, yg, xc, yc)
    if xg is None and xc is None:
        return False

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(6.4, 5.0), dpi=150,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
        sharex=True
    )

    # Top: overlay
    if xg is not None:
        ax.plot(xg, yg, label="Guinea-Pig", lw=1.2)
    if xc is not None:
        if yc_on_gp is not None:
            ax.plot(xg, yc_on_gp, label="Circe2", lw=1.2)
        else:
            ax.plot(xc, yc, label="Circe2", lw=1.2)
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel("N")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.2)
    _draw_collider_label(ax, collider_label)

    # Bottom: ratio (Circe2 / GP) on GP grid
    ratio = None
    if xg is not None and yg is not None and yc_on_gp is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(yg > 0, yc_on_gp / yg, np.nan)
    if ratio is not None:
        rax.plot(xg, ratio, color="tab:blue", lw=1.0)
        rax.axhline(1.0, color="gray", lw=0.8, ls="--")
        rax.set_ylabel("C2/GP")
        if ratio_ylim and len(ratio_ylim) == 2:
            rax.set_ylim(ratio_ylim[0], ratio_ylim[1])
        rax.grid(alpha=0.2)
    rax.set_xlabel(title.split()[0])

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return True

def plot_2d(gp_path, c2_path, title, out_path, collider_label: str = ""):
    xg, yg, Zg = load_2d(gp_path)
    xc, yc, Zc = load_2d(c2_path)
    if Zg is None and Zc is None:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), dpi=150, constrained_layout=True)
    if Zg is not None:
        im = axes[0].pcolormesh(xg, yg, Zg, shading="auto")
        axes[0].set_title("Guinea-Pig")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    if Zc is not None:
        im = axes[1].pcolormesh(xc, yc, Zc, shading="auto")
        axes[1].set_title("Circe2")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(title, y=1.02)
    # Put collider label on the left panel
    _draw_collider_label(axes[0], collider_label)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return True

def main():
    ap = argparse.ArgumentParser(description="Overlay Circe2 vs GP histograms with ratio panel for 1D.")
    ap.add_argument("--dir", default=".", help="Directory with hist files")
    ap.add_argument("--out-prefix", default="", help="Prefix for output filenames")
    ap.add_argument("--format", default="pdf", choices=["pdf","png","svg"], help="Output format")
    ap.add_argument("--channels", nargs="+", default=["ee","eg","ge","gg"])
    ap.add_argument("--no-logy", action="store_true", help="Disable log scale on Y in the top panel")
    ap.add_argument("--ratio-ylim", nargs=2, type=float, metavar=("LOW","HIGH"), help="Fix ratio y-limits, e.g. 0.5 1.5")
    ap.add_argument("--collider-label", default="", help="Optional collider label text. Defaults to a label derived from --out-prefix.")
    args = ap.parse_args()

    hist1d = [("x","x"), ("y","y")]
    hist2d = [("xy","x,y"), ("x-y","x-y")]

    # Determine collider label: explicit override wins; else derive from out-prefix
    collider_label = args.collider_label.strip() if args.collider_label else _format_label_from_prefix(args.out_prefix)

    for ch in args.channels:
        # 1D with ratio
        for h, xlabel in hist1d:
            gp = os.path.join(args.dir, f"{h}.{ch}.gp")
            c2 = os.path.join(args.dir, f"{h}.{ch}.circe2")
            out = os.path.join(args.dir, f"{args.out_prefix+'.' if args.out_prefix else ''}{h}.{ch}.{args.format}")
            plot_1d_with_ratio(
                gp, c2, f"{h.upper()} ({ch})", out,
                logy=not args.no_logy,
                ratio_ylim=tuple(args.ratio_ylim) if args.ratio_ylim else None,
                collider_label=collider_label
            )
        # 2D unchanged (no ratio panel)
        for h, _ in hist2d:
            gp = os.path.join(args.dir, f"{h}.{ch}.gp")
            c2 = os.path.join(args.dir, f"{h}.{ch}.circe2")
            out = os.path.join(args.dir, f"{args.out_prefix+'.' if args.out_prefix else ''}{h}.{ch}.{args.format}")
            plot_2d(gp, c2, f"{h.upper()} ({ch})", out, collider_label=collider_label)

if __name__ == "__main__":
    main()
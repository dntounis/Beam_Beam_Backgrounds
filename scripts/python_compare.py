#!/usr/bin/env python3
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    # Expect 3 columns: x y z
    try:
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data.reshape(-1, 3)
        if data.shape[1] < 3:
            return None, None, None
        xs = np.unique(data[:, 0])
        ys = np.unique(data[:, 1])
        # map to grid
        Z = np.full((ys.size, xs.size), np.nan)
        # rows are not guaranteed sorted; build index
        x_index = {v: i for i, v in enumerate(xs)}
        y_index = {v: i for i, v in enumerate(ys)}
        for x, y, z in data:
            Z[y_index[y], x_index[x]] = z
        return xs, ys, Z
    except Exception:
        return None, None, None

def plot_1d(gp_path, c2_path, title, out_path, logy=True):
    xg, yg = load_1d(gp_path)
    xc, yc = load_1d(c2_path)
    if xg is None and xc is None:
        return False
    plt.figure(figsize=(6,4), dpi=150)
    if xg is not None:
        plt.plot(xg, yg, label="Guinea-Pig", lw=1.2)
    if xc is not None:
        plt.plot(xc, yc, label="Circe2", lw=1.2)
    if logy:
        plt.yscale("log")
    plt.xlabel(title.split()[0])
    plt.ylabel("N")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True

def plot_2d(gp_path, c2_path, title, out_path):
    xg, yg, Zg = load_2d(gp_path)
    xc, yc, Zc = load_2d(c2_path)
    if Zg is None and Zc is None:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(8,3.6), dpi=150, constrained_layout=True)
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
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return True

def main():
    ap = argparse.ArgumentParser(description="Overlay Circe2 vs GP histograms")
    ap.add_argument("--dir", default=".", help="Directory with hist files")
    ap.add_argument("--out-prefix", default="", help="Prefix for output filenames")
    ap.add_argument("--format", default="pdf", choices=["pdf","png","svg"], help="Output format")
    ap.add_argument("--channels", nargs="+", default=["ee","eg","ge","gg"])
    args = ap.parse_args()

    hist1d = [("x","x"), ("y","y")]
    hist2d = [("xy","x,y"), ("x-y","x-y")]
    for ch in args.channels:
        # 1D overlays
        for h, xlabel in hist1d:
            gp = os.path.join(args.dir, f"{h}.{ch}.gp")
            c2 = os.path.join(args.dir, f"{h}.{ch}.circe2")
            out = os.path.join(args.dir, f"{args.out_prefix+'.' if args.out_prefix else ''}{h}.{ch}.{args.format}")
            plot_1d(gp, c2, f"{h.upper()} ({ch})", out, logy=True)
        # 2D side-by-sides
        for h, _ in hist2d:
            gp = os.path.join(args.dir, f"{h}.{ch}.gp")
            c2 = os.path.join(args.dir, f"{h}.{ch}.circe2")
            out = os.path.join(args.dir, f"{args.out_prefix+'.' if args.out_prefix else ''}{h}.{ch}.{args.format}")
            plot_2d(gp, c2, f"{h.upper()} ({ch})", out)

if __name__ == "__main__":
    main()

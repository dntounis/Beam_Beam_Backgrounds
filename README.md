### Pair Envelope Analysis and Plotting

This directory contains a Python tool to compute and visualize trajectory envelopes of beam–induced e⁺/e⁻ pairs in a solenoidal magnetic field. It reads GuineaPig-style output files (`pairs_*.dat`), propagates particles analytically as helices, fills 2D histograms (z vs r/x/y), and derives percentile envelope curves. It can optionally overlay detector barrel layers and plot a log-scaled density map.

The main entry point is `pair_envelope_v4.py`.

An example envelope plot is included in this repository for reference.

## Features
- **Fast helix propagation**: Analytical trajectory computation with Numba acceleration
- **2D histograms**: z vs r, x, or y with log color scaling
- **Percentile envelopes**: e.g. 68%, 95%, 99%, 99.9%, 99.99%
- **Optional per-bunch normalization**: Normalize aggregated histograms back to a single bunch via `--normalize-per-bunch`
- **Parallel processing**: Multi-process across many `.dat` files
- **Result caching**: Saves/loads aggregated histograms and trajectories to speed up reruns
- **Plot overlays**: Beampipe lines and silicon barrel layers
- **Density units toggle**: Choose counts per bin or counts per mm²
- **Zero bins shown white**: Empty histogram bins render as white for clarity
- **ROOT kBird colormap**: `--cmap root` reproduces ROOT's kBird palette (via provided stops/RGB)
  (and any bins below the colorbar minimum are also shown white)
-- **pT–θ diagnostics and ridge extraction**: Saves auxiliary pT–θ plots and can extract/save a deflection ridge for reuse.
- **Reachability boundary overlay**: Automatically computes and overlays detector reachability boundaries on pT–θ plots using the companion `reachability_analysis.py` script.

## Requirements
- Python 3.8+
- Packages: `numpy`, `matplotlib`, `tqdm`, `numba`, `mplhep` (optional, for CMS style)

Install (example):
```bash
pip install numpy matplotlib tqdm numba mplhep
```

## Input data format
- The script scans `--indir` for files named like `pairs_*.dat`.
- Each line in a `.dat` file should have at least 7 whitespace-separated columns:
  1. energy [GeV]
  2. beta_x [unitless]
  3. beta_y [unitless]
  4. beta_z [unitless]
  5. vtx_x [nm]
  6. vtx_y [nm]
  7. vtx_z [nm]
- Any extra columns are ignored.

## Quick start

### Basic envelope analysis
Example invocation:
```bash
python pair_envelope_v4.py \
  --indir /path/to/bkg_particles \
  --max-files 266 \
  --plot-trajectories 20 \
  --num-bunches 266 \
  --draw-2d-histo \
  --field 5.0 \
  --nz 400 --nr 120 \
  --zmin -300 \
  --zmax +300 --rmin -30 --rmax 30 \
  --percentiles 68 95 99 99.9 99.99 \
  --coord r \
  --out envelope_r_vs_z.pdf \
  --pt-cut 0.0 \
  --collider "C³ 250 PS1" \
  --detector "SiD_o2_v04" \
  --smooth-envelopes --show-detector \
  --parallel --parallel-threads 15 \
  --use-numba \
  --cache-trajectory-data \
  --show-colorbar --colorbar-min 1 --colorbar-max 1e7 \
  --density-units per_mm2 \
  --normalize-per-bunch \
  --cmap root
```

### High-resolution analysis with reachability boundary
For detailed studies requiring higher resolution and detector reach analysis:
```bash
python pair_envelope_v4.py \
  --indir /path/to/bkg_particles \
  --max-files 200 \
  --num-bunches 200 \
  --draw-2d-histo \
  --field 5.0 \
  --nz 400 --nr 200 \
  --zmin -300 --zmax 300 \
  --rmin -29 --rmax 29 \
  --percentiles 68 95 99 99.9 99.97 \
  --coord r \
  --out envelope_r_vs_z_high_res.pdf \
  --pt-cut 0.0 \
  --collider "C³ 250 PS1" \
  --detector "SiD_o2_v04" \
  --smooth-envelopes --show-detector \
  --parallel --parallel-threads 40 \
  --use-numba \
  --cache-trajectory-data \
  --show-colorbar --colorbar-min 1 --colorbar-max 1e5 \
  --density-units per_mm2 \
  --normalize-per-bunch \
  --cmap root \
  --roi-theta-min 5e-3 --roi-pt-min 1e-3 \
  --reachability-script reachability_analysis.py
```

### Multiple collider configuration comparison
For comparing different collider setups:
```bash
# C³ 250 GeV PS1 configuration
python pair_envelope_v4.py \
  --indir /path/to/C3_250_PS1/bkg_particles \
  --max-files 200 --num-bunches 200 \
  --draw-2d-histo --field 5.0 \
  --nz 400 --nr 200 \
  --zmin -300 --zmax 300 --rmin -29 --rmax 29 \
  --percentiles 68 95 99 99.9 99.97 \
  --coord r --out envelope_C3_250_PS1.pdf \
  --collider "C³ 250 PS1" --detector "SiD_o2_v04" \
  --smooth-envelopes --show-detector \
  --parallel --parallel-threads 40 \
  --use-numba --cache-trajectory-data \
  --show-colorbar --colorbar-min 1 --colorbar-max 1e5 \
  --density-units per_mm2 --normalize-per-bunch \
  --cmap root --roi-theta-min 5e-3 --roi-pt-min 1e-3

# C³ 550 GeV PS1 configuration  
python pair_envelope_v4.py \
  --indir /path/to/C3_550_PS1/bkg_particles \
  --max-files 200 --num-bunches 200 \
  --draw-2d-histo --field 5.0 \
  --nz 400 --nr 200 \
  --zmin -300 --zmax 300 --rmin -29 --rmax 29 \
  --percentiles 68 95 99 99.9 99.97 \
  --coord r --out envelope_C3_550_PS1.pdf \
  --collider "C³ 550 PS1" --detector "SiD_o2_v04" \
  --smooth-envelopes --show-detector \
  --parallel --parallel-threads 40 \
  --use-numba --cache-trajectory-data \
  --show-colorbar --colorbar-min 1 --colorbar-max 1e5 \
  --density-units per_mm2 --normalize-per-bunch \
  --cmap root --roi-theta-min 3e-3 --roi-pt-min 1e-3
```

This will:
- Read up to 266 `pairs_*.dat` files from `--indir`
- Aggregate histograms across files and (with `--normalize-per-bunch`) normalize to a single bunch crossing
- Compute envelopes at the given percentiles
- Draw a log-scaled 2D histogram with a colorbar
- Save the plot to `--out`
- Additionally, auxiliary pT–θ plots are saved next to `--out` and, if requested, a JSON file with the extracted deflection ridge (see options below).
- Automatically overlay reachability boundaries on pT–θ plots (unless disabled with `--no-reachability-boundary`)

## Argument reference
- **--indir DIR** (required): Directory containing input `.dat` files. Files are matched with the pattern `pairs_*.dat`. Each file is treated as one bunch crossing when normalizing.

- **--max-files N**: Maximum number of files to process from `--indir` after sorting. Used for both processing and, on fresh runs, to derive the per-bunch normalization (histograms are divided by the number of files processed).

- **--field B**: Magnetic field in Tesla. Sets the solenoidal field used in helix propagation.

- **--nz N**: Number of z-bins.
- **--nr N**: Number of r-bins.

- **--zmin ZMIN**, **--zmax ZMAX**: z-range in mm for tracking and histogramming.
- **--rmin RMIN**, **--rmax RMAX**: r-range in mm for tracking and histogramming and for plot extents.

- **--percentiles P [P ...]**: Space-separated list of percentiles to compute envelope curves for (e.g. `68 95 99 99.9 99.99`).

- **--coord {r,x,y}**: Which coordinate to analyze/plot versus z. Affects both the 2D histogram and the envelope curves drawn.

- **--out PATH** (required): Output image file for the plot. The extension determines the format (e.g. `.pdf`, `.png`).

- **--draw-2d-histo**: If set, draws the 2D histogram as a log-scaled colormap behind the envelopes. If omitted, you can optionally overlay individual trajectories instead.

- **--parallel**: Enable multi-process processing of input files.

- **--save-envelopes PATH**: If provided, saves the computed envelopes as a pickle file at `PATH`. The object is a dict keyed by percentile with values `(z_centers, envelope_pos, envelope_neg)`.

- **--plot-trajectories N**: Number of individual trajectories to overlay. Only plotted when `--draw-2d-histo` is NOT used (to avoid clutter). Trajectories are kept up to an internal cap for performance.

- **--pt-cut PT**: pT selection threshold in GeV/c. Only particles with pT > PT contribute to the “passes selection” counts and optional trajectory overlay. Defaults to `4e-3`.

- **--roi-theta-min THETA_MIN**: Minimum theta (rad) for the deflection ridge ROI. Default `2e-3`.
- **--roi-theta-max THETA_MAX**: Maximum theta (rad) for the deflection ridge ROI (omit for no upper bound).
- **--roi-pt-min PT_MIN**: Minimum pT (GeV/c) for the deflection ridge ROI. Default `1e-3`.
- **--roi-pt-max PT_MAX**: Maximum pT (GeV/c) for the deflection ridge ROI (omit for no upper bound).

- **--smooth-envelopes**: Apply simple moving-average smoothing to the positive/negative envelope curves.

- **--collider NAME**: Text label for the collider parameter set (displayed on the plot title). Default: `C³ 250 PS1`.

- **--num-bunches N**: Text label for the number of bunches (displayed on the plot title). Also used as a fallback normalization factor when loading cached results that predate embedded metadata.

- **--detector NAME**: Detector configuration label printed on the plot. Default: `SiD_o2_v04`.

- **--show-detector**: If set, overlays silicon vertex barrel layers on the r–z view (only meaningful for `--coord r`).

- **--use-numba**: Use Numba acceleration. The script is Numba-accelerated by default; this flag is provided for clarity/compatibility.

- **--cache-trajectory-data**: If set, the script will cache aggregated results to a pickle file inside `--indir` and reuse it on subsequent runs with compatible settings. The cache includes the number of files aggregated; upon load, histograms are normalized to one bunch by dividing by that number. If that metadata is missing, `--num-bunches` is used as a fallback normalization factor.

- **--parallel-threads N**: Number of worker processes for `--parallel`. Defaults to all available cores.

- **--show-colorbar**: If set, shows a colorbar for the 2D histogram. The label matches `--density-units`.

- **--colorbar-min V**, **--colorbar-max V**: Log color scale bounds (vmin/vmax) for the histogram when `--draw-2d-histo` is used. If `--colorbar-max` is omitted, vmax auto-scales to data.

- **--density-units {per_bin, per_mm2}**: Units for the 2D color map.
  - `per_bin` (default): counts per bin per bunch; colorbar label “Tracks/bin”.
  - `per_mm2`: counts per mm² per bunch; divides by bin area Δz×Δr (mm²) computed from `--zmin/--zmax/--nz` and `--rmin/--rmax/--nr`. Colorbar label “Tracks/mm²”.

- **--normalize-per-bunch**: If set, divide aggregated histograms by the number of input files (or cached `num_files`) so values correspond to a single bunch crossing. The title displays “(1 bunch)” (singular). If not set, values reflect the aggregate over all processed files and the title shows the aggregated bunch count.

- **--cmap {viridis,turbo,plasma,inferno,magma,root}**: Colormap for the 2D histogram. `root` reproduces ROOT's kBird palette using the official gradient definition.

- **--save-deflection-ridge PATH**: Save deflection ridge data to JSON. If omitted, a default path is derived by appending `_deflection_ridge.json` next to `--out`.

- **--no-reachability-boundary**: Disable the reachability boundary overlay on pT–θ plots (enabled by default).

- **--reachability-script PATH**: Path to the reachability analysis script. Defaults to `reachability_analysis.py` in the same directory.

- **--reachability-pt-min PT**: Minimum pT (GeV/c) to sample reachability boundary. Defaults to computed threshold.

- **--reachability-pt-max PT**: Maximum pT (GeV/c) to sample reachability boundary. Defaults to plot upper bound.

- **--reachability-pt-samples N**: Number of pT samples for reachability boundary. Default: 400.

- **--reachability-linear-pt**: Use linear spacing for reachability boundary sampling (default is log spacing).

- **--reachability-charge Q**: Effective charge factor q for reachability boundary (GeV·T·mm). Default: 0.3.

- **--reachability-mag-field B**: Magnetic field B0 in Tesla for reachability boundary. Default: 5.0.

- **--reachability-detector-radius R**: Detector radius in millimetres for reachability boundary. Default: 14.0.

- **--reachability-z-max Z**: Detector z-extent in millimetres for reachability boundary. Default: 76.0.

- **--reachability-theta-upper THETA**: Optional theta upper bound (rad) when sampling reachability boundary.

## Outputs
- **Plot**: Saved to `--out`. Shows envelope curves and (optionally) the log-scaled 2D histogram, beampipe boundary lines, and detector layer overlays. x-axis is z [mm]; y-axis is `r`/`x`/`y` [mm].
- **Envelopes (optional)**: Pickled dict if `--save-envelopes` is provided.
- **Cache (optional)**: When `--cache-trajectory-data` is used, a file named like `cached_results_{B}T_{nz}x{nr}_{zmin}to{zmax}_{max_files}_bunches.pkl` is written in `--indir`. It contains aggregated histograms, a subset of trajectories (if kept), and counters. On load, histograms are normalized per bunch.

- **pT–θ scatter**: `<out>_pt_theta.png`.
- **pT–θ 2D histogram**: `<out>_pt_theta_hist2d.pdf` (log–log axes).
- **pT–θ 2D histogram with deflection ridge**: `<out>_pt_theta_hist2d_deflection.pdf` (includes ridge and optional fit overlay).
- **Deflection ridge JSON (optional)**: If `--save-deflection-ridge` is set (or by default next to `--out`), a JSON with arrays `ridge_theta`, `ridge_pt` and optional `line_theta`, `line_pt`, plus an optional `power_law` tuple and metadata.

## Notes and tips
- For per-bunch plots, set `--max-files` to the number of bunches you aggregate and pass `--normalize-per-bunch`.
- Use `--parallel` with `--parallel-threads` to speed up large directories.
- Enable `--cache-trajectory-data` to avoid recomputing the propagation and histogramming on iterative plotting tweaks.
- Trajectories are drawn only when `--draw-2d-histo` is not set, to keep the visualization readable.
- Bins with values strictly below `--colorbar-min` (or ≤0) are masked and rendered white to emphasize the dynamic range above the threshold.
- The y-axis limits and the color map extent follow `--rmin/--rmax`; choose these alongside `--nr` for your target radial resolution.
- The reachability boundary overlay requires `reachability_analysis.py` to be present in the same directory or specified via `--reachability-script`.
- Reachability boundary parameters should match your detector geometry and magnetic field settings for accurate results.

## Deflection ridge overlay tool (`pair_envelope_deflection_ridge.py`)

This companion script overlays one or more deflection ridge curves (JSON saved by `pair_envelope_v4.py`) on a single pT–θ log–log plot. It can also restrict the view to a region-of-interest, draw detector reach boundaries, optionally overlay a 2D histogram from cached PKL data, and report acceptance fractions.

### Basic deflection ridge comparison
```bash
python pair_envelope_deflection_ridge.py \
  --ridge-250-PS1 /path/to/C3_250_PS1_deflection_ridge.json \
  --ridge-250-PS2 /path/to/C3_250_PS2_deflection_ridge.json \
  --ridge-550-PS1 /path/to/C3_550_PS1_deflection_ridge.json \
  --ridge-550-PS2 /path/to/C3_550_PS2_deflection_ridge.json \
  --pkl-250-PS1 /path/to/cache_250_PS1.pkl \
  --pkl-250-PS2 /path/to/cache_250_PS2.pkl \
  --pkl-550-PS1 /path/to/cache_550_PS1.pkl \
  --pkl-550-PS2 /path/to/cache_550_PS2.pkl \
  --roi-theta-min 2e-3 --roi-pt-min 1e-3 \
  --pt-det-min 2e-3 --theta-det-min 3e-3 \
  --overlay-hist 250_PS1 --overlay-alpha 0.85 --overlay-colorbar \
  --show-fit-line \
  --out deflection_ridges_overlay.png
```

### Advanced deflection ridge analysis with acceptance computation
For comprehensive analysis including detector acceptance fractions:
```bash
python pair_envelope_deflection_ridge.py \
  --ridge-250-PS1 /path/to/C3_250_PS1_deflection_ridge.json \
  --ridge-250-PS2 /path/to/C3_250_PS2_deflection_ridge.json \
  --ridge-550-PS1 /path/to/C3_550_PS1_deflection_ridge.json \
  --ridge-550-PS2 /path/to/C3_550_PS2_deflection_ridge.json \
  --pkl-250-PS1 /path/to/cache_250_PS1.pkl \
  --pkl-250-PS2 /path/to/cache_250_PS2.pkl \
  --pkl-550-PS1 /path/to/cache_550_PS1.pkl \
  --pkl-550-PS2 /path/to/cache_550_PS2.pkl \
  --pt-det-min 1.05e-2 --theta-det-min 0.77 \
  --roi-theta-min 1e-2 --roi-theta-max 1.571 \
  --roi-pt-min 1e-3 --roi-pt-max 0.5 \
  --overlay-hist 250_PS1 --overlay-colorbar \
  --reachability-script reachability_analysis.py \
  --out pt_theta_ridges_with_overlay.pdf \
  --hide-overlay-hist
```

Key options:
- `--ridge-XXX`: Input ridge JSONs containing `ridge_theta`/`ridge_pt` (and optionally the fitted line and power-law).
- `--pkl-XXX`: Optional PKL caches to enable density overlay and acceptance fraction computation.
- `--overlay-hist {250_PS1,250_PS2,550_PS1,550_PS2}`: Which dataset’s histogram to overlay (requires corresponding `--pkl-...`).
- `--roi-theta-min/max`, `--roi-pt-min/max`: Visual ROI guides and axis limits when provided.
- `--pt-det-min`, `--theta-det-min`: Detector reach cuts; draws dashed boundaries and a translucent accepted region.
- `--show-fit-line`: If the JSONs include a fit, draw it and annotate the power-law.

The script prints acceptance fractions to stdout when PKL inputs are available and saves the requested overlay plot.

## Reachability analysis tool (`reachability_analysis.py`)

This utility script computes and visualizes the detector reachability boundary in the pT–θ plane. It determines the minimum pT required for particles to reach the detector barrel at different angles, accounting for the magnetic field and detector geometry.

### Usage
```bash
python reachability_analysis.py \
  --mag-field 5.0 \
  --detector-radius 14.0 \
  --z-max 76.0 \
  --charge 0.3 \
  --pt-min 1e-3 \
  --pt-max 0.1 \
  --pt-samples 200 \
  --theta-upper 0.1 \
  --out reachability_boundary.pdf
```

Key options:
- `--mag-field B`: Magnetic field strength in Tesla. Default: 5.0.
- `--detector-radius R`: Detector barrel radius in mm. Default: 14.0.
- `--z-max Z`: Detector z-extent in mm. Default: 76.0.
- `--charge Q`: Effective charge factor (GeV·T·mm). Default: 0.3.
- `--pt-min PT`: Minimum pT for sampling (GeV/c). Default: computed threshold.
- `--pt-max PT`: Maximum pT for sampling (GeV/c). Default: 0.1.
- `--pt-samples N`: Number of pT samples. Default: 200.
- `--linear-pt`: Use linear spacing instead of log spacing.
- `--theta-upper THETA`: Upper theta bound in radians. Default: 0.1.
- `--out PATH`: Output plot file path.

The script computes the reachability boundary and saves it as a plot showing the minimum pT required to reach the detector as a function of angle.


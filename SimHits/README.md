## EDM4hep Timing Studies Snapshot

This directory provides a  Python workflow to analyze EDM4hep timing information and to produce background hit-rate summaries for electron-positorn collider studies. Every tool here can be run directly with:

```
python timing_studies/<script_name>.py [options]
```

All outputs (plots, `.npz` caches, JSON reports) are written relative to the working directory from which the command is launched. The scripts share a common set of helper routines and can be mixed and matched as needed for detector studies or documentation.

---

## Installation

1. Ensure Python 3.9 or newer is available.
2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
   ```
3. Install required Python packages:
   ```bash
pip install uproot awkward numpy matplotlib
   ```
4. Install optional helpers (recommended for full functionality):
   ```bash
pip install scipy mplhep
```

---

## Included Tools

- `timing_studies/timing_all_BX_Aug2025.py`
  - Multi–bunch-crossing timing histogrammer with IPC/HPP separation, rate extraction, and tracker-layer visualisations.
- `timing_studies/timing_one_BX_Aug2025.py`
  - Per-BX timing analysis that overlays IPC and HPP backgrounds, supports branch-specific plotting windows, and normalises counts per bunch crossing.
- `timing_studies/hit_rates_plot_Aug2025.py`
  - Generates the combined-background hit-rate figures and writes sensitive-area metadata to JSON.
- `timing_studies/hit_rates_plot_Aug2025_separate_IPC_HPP.py`
  - Extends the hit-rate workflow with individual IPC and HPP plots and tabulated summaries.
- `timing_studies/hit_plotting.ipynb`
  - Notebook interface mirroring the script APIs for exploratory visualisation (IPC-only or IPC+HPP).

---

## Data Flow Overview

1. **Input discovery** – EDM4hep ROOT files are located using glob patterns (default `ddsim_C3_250_PS1_v2_seed_*.edm4hep.root`). Seeds are parsed from `seed_####` and sorted numerically.
2. **Event processing** – `uproot` and `awkward` read the ragged hit collections. Trackers rely on `<branch>.time` (with optional `<branch>.eDep` thresholds); calorimeters use `<branch>Contributions.time` with positional filters and optional energy thresholds.
3. **Histogramming** – Times are accumulated into base histograms defined by `--time-min`, `--time-max`, and `--base-binwidth`. Multi-BX studies offset each input file by `ifile * bunch_spacing`.
4. **Normalisation** – IPC backgrounds average over the number of seed files. HPP backgrounds use `(hpp_mu / num_events)` scaling and can be replicated across bunches.
5. **Plotting & caching** – Re-binned spectra (`--plot-binwidth`) feed publication-quality plots. All intermediate data are written to `.npz` caches for fast reuse.

---

## Multi-Bunch Crossing Analysis (`timing_all_BX_Aug2025.py`)

### Key Features

- Processes IPC samples spanning multiple bunch crossings by offsetting each seed file in time.
- Stores three histogram families per branch: total, IPC-only, and HPP-only (`counts_<branch>`, `counts_ipc_<branch>`, `counts_hpp_<branch>`).
- Tracks raw hit counts per branch (`raw_entries_<branch>`) to cross-check normalisation.
- Optional HPP handling through `--hpp-file` and `--hpp-mu`, including replication across bunches via `accumulate_hpp_counts`.
- Constant-fit rate extraction; when `--rates-json` is provided the results are merged into a nested JSON organised by location (barrel/endcap) and detector category.
- Tracker overlay plots and SiVertexBarrel layer studies (`--vb-layers` or `--vb-layers-root`).

### Primary Arguments

- `--base-dir DIR` (required): location of IPC EDM4hep files.
- `--pattern GLOB`: filename pattern (default `ddsim_C3_250_PS1_v2_seed_*.edm4hep.root`).
- `--bunch-spacing NS`: bunch spacing in nanoseconds (default 5.25).
- `--time-min NS`, `--time-max NS`: histogram bounds (default 0 to 800 ns).
- `--base-binwidth NS`: base histogram bin width (default 0.25 ns).
- `--plot-binwidth NS`: plotting bin width; must be an integer multiple of `--base-binwidth` (default 10 ns).
- `--out FILE`: `.npz` cache (default `timing_all_BX_histograms.npz`).
- `--outdir DIR`: destination for PNG/PDF plots and optional JSON.
- `--use-cache`: reuse an existing `.npz` if configuration and file list match.
- `--load-hist FILE`: load an existing `.npz` and skip ROOT file processing.
- `--rates-json FILE`: write IPC/HPP/Total rates to JSON (merged with existing content when possible).
- Energy thresholds (per collection):
  - `--thr-vertex-barrel`, `--thr-vertex-endcap`, `--thr-tracker-barrel`, `--thr-tracker-endcap`, `--thr-tracker-forward`,
  - `--thr-ecal-barrel`, `--thr-ecal-endcap`, `--thr-hcal-barrel`, `--thr-hcal-endcap`,
  - `--thr-muon-barrel`, `--thr-muon-endcap`, `--thr-lumical`, `--thr-beamcal`.
- Visual controls: `--tracker-overlay`, `--overlay-ymax`, `--vb-layers`, `--vb-ymax`, `--vb-layers-root`, `--legend-fs`.

### Outputs

- Per-branch timing plots (PNG/PDF) with constant-fit overlays and 68% confidence bands.
- Optional tracker overlay and SiVertexBarrel layer figures (hits/ns, hits, hits/(ns·mm²), hits/mm²).
- `.npz` cache containing bin edges, file list, IPC/HPP/total histograms, raw hit counters, and (if HPP enabled) per-BX HPP histograms (`hpp_counts_per_bx_<branch>`).
- Optional rates JSON summarising constant-fit yields.

---

## Single-Bunch Crossing Analysis (`timing_one_BX_Aug2025.py`)

### Key Features

- Averages IPC data per bunch crossing without applying time offsets.
- Supports optional HPP overlays with proper per-BX scaling (`hpp_counts_<branch>` stored alongside IPC counts).
- Allows branch-specific plotting windows via `--tmax-<collection>` arguments so late-hit tails can be trimmed independently.
- `--show-mean` adds constant-fit bands for both IPC and HPP; `--logy` toggles logarithmic y-axes.

### Primary Arguments

- `--base-dir DIR` (required): IPC EDM4hep directory.
- `--pattern GLOB`: IPC file pattern (same default as multi-BX script).
- `--seeds LIST`: space-separated list of seed numbers to process (optional).
- `--max-files N`: cap on IPC files after seed filtering (optional).
- `--time-min NS`, `--time-max NS`: histogram bounds (default 0 to 800 ns).
- `--base-binwidth NS`, `--plot-binwidth NS`: accumulation and plotting bin widths.
- `--out FILE`: `.npz` cache (default `timing_one_BX_histograms.npz`).
- `--outdir DIR`: destination for plots.
- `--use-cache`, `--load-hist`: same semantics as the multi-BX script.
- `--hpp-file FILE`: many-event HPP EDM4hep sample.
- `--hpp-mu VALUE`: expected HPP events per BX (required when `--hpp-file` is used to enable proper scaling).
- `--legend-fs`: legend font size.
- `--show-mean`, `--logy`: plotting toggles.
- Branch-specific plot limits:
  - `--tmax-vertex-barrel`, `--tmax-vertex-endcap`, `--tmax-tracker-barrel`, `--tmax-tracker-endcap`, `--tmax-tracker-forward`,
  - `--tmax-ecal-barrel`, `--tmax-ecal-endcap`, `--tmax-hcal-barrel`, `--tmax-hcal-endcap`,
  - `--tmax-muon-barrel`, `--tmax-muon-endcap`, `--tmax-lumical`, `--tmax-beamcal`.
- Energy thresholds: same set as the multi-BX script.

### Outputs

- Dual-background plots per branch (`<collider>_one_BX_timing_<branch>.(png|pdf)`) with IPC (blue) and HPP (orange) curves.
- `.npz` cache storing IPC counts, HPP counts, bin edges, processed file list, and HPP metadata (`hpp_file`, `hpp_mu`).

---

## Hit-Rate Summary Plots

### `hit_rates_plot_Aug2025.py`

- Recreates barrel and endcap hit-rate figures using geometry-derived sensitive areas.
- Writes area metadata (mm², cm², detector labels, assumptions) to JSON unless `--skip-area-density` is supplied.
- Produces both absolute hit-rate and surface-density (hits/(ns·cm²)) figures.

### `hit_rates_plot_Aug2025_separate_IPC_HPP.py`

- Extends the base workflow with dedicated IPC and HPP plots and tabulated CSV/JSON summaries in the output directory.
- Supports the same geometry reconstruction pipeline and plotting controls as the combined script.

### Shared Arguments

- `--outdir DIR`: destination for plots and reports.
- `--formats ext1 ext2 ...`: one or more output formats (e.g. `png pdf`).
- `--dpi VALUE`: rasterisation resolution for PNG outputs.
- `--no-mplhep`: disable CMS-style theming if `mplhep` is installed.
- `--area-report FILE`: write sensitive-area metadata to a custom path.
- `--title-prefix`, `--title-suffix`: append text to figure titles.
- Optional energy-threshold JSON inputs can be supplied via dedicated script arguments when needed.

---

## Units and Conventions

- Histograms accumulate counts per `--base-binwidth` (raw hits per bin).
- Plots convert counts to densities by dividing by `--plot-binwidth`, yielding hits/ns (or hits/(ns·mm²) for area-normalised plots).
- Constant-fit means exclude the first and last 5 ns of the full histogram range to avoid edge effects.
- Calorimeter hits use the first contribution time per hit; energy thresholds are evaluated on that contribution when available.

---

## Outputs at a Glance

- Timing scripts:
  - Per-collection PNG/PDF plots, tracker overlays, SiVertexBarrel layer figures.
  - `.npz` caches containing histograms, bin edges, file lists, raw hit counts, and (when applicable) HPP metadata.
  - Optional JSON summaries of constant-fit rates.
- Hit-rate scripts:
  - Barrel/endcap (and IPC/HPP) plots in requested formats.
  - Geometry-derived area reports (`detector_areas.json` by default).
  - Optional CSV/JSON summaries for IPC/HPP splits.

---

## Example Usage

```bash
# Multi-BX timing with HPP overlay and tracker overlay plots
python timing_studies/timing_all_BX_Aug2025.py \
  --base-dir /data/ipc \
  --pattern "ddsim_C3_250_PS1_v2_seed_*.edm4hep.root" \
  --hpp-file /data/hpp/hpp_background.edm4hep.root \
  --hpp-mu 0.12 \
  --bunch-spacing 5.25 \
  --plot-binwidth 10 \
  --tracker-overlay \
  --vb-layers \
  --outdir outputs/timing_all

# Per-BX timing with reduced ECAL time window and log-scale plots
python timing_studies/timing_one_BX_Aug2025.py \
  --base-dir /data/ipc \
  --hpp-file /data/hpp/hpp_background.edm4hep.root \
  --hpp-mu 1.8 \
  --tmax-ecal-endcap 400 \
  --show-mean \
  --logy \
  --outdir outputs/timing_one

# Hit-rate figures with area-density outputs
python timing_studies/hit_rates_plot_Aug2025.py \
  --outdir outputs/hit_rates \
  --formats png pdf \
  --area-report outputs/hit_rates/detector_areas.json

# Split IPC/HPP hit-rate figures without mplhep styling
python timing_studies/hit_rates_plot_Aug2025_separate_IPC_HPP.py \
  --outdir outputs/hit_rates_split \
  --formats png pdf \
  --no-mplhep

# Batch per-BX sweep across collider scenarios
for scenario in C3_250_PS1 C3_250_PS2 C3_550_PS1 C3_550_PS2; do
  base="/data/GuineaPig_runs/${scenario}/ddsim"
  hpp="/data/hpp/${scenario}/gg_had_MERGED.edm4hep.root"
  tag="$(echo ${scenario} | tr '[:upper:]' '[:lower:]')"
  python timing_studies/timing_one_BX_Aug2025.py \
    --base-dir "${base}" \
    --pattern "ddsim_${scenario}_v2_seed_*" \
    --max-files 1800 \
    --time-min 0 --time-max 700 \
    --tmax-vertex-barrel 60 --tmax-vertex-endcap 90 \
    --tmax-tracker-barrel 160 --tmax-tracker-endcap 110 --tmax-tracker-forward 85 \
    --tmax-ecal-barrel 160 --tmax-ecal-endcap 210 \
    --tmax-hcal-barrel 720 --tmax-hcal-endcap 720 \
    --tmax-muon-barrel 45 --tmax-muon-endcap 720 \
    --base-binwidth 0.25 \
    --plot-binwidth 6.25 \
    --out "timing_one_BX_${scenario}_scan.npz" \
    --outdir "outputs/one_BX_${tag}_6p25" \
    --thr-vertex-barrel 6.0e-7 --thr-vertex-endcap 5.0e-7 \
    --thr-tracker-barrel 3.0e-5 --thr-tracker-endcap 3.0e-5 --thr-tracker-forward 4.0e-7 \
    --thr-ecal-barrel 5.0e-5 --thr-ecal-endcap 5.0e-5 \
    --thr-hcal-barrel 2.4e-4 --thr-hcal-endcap 2.3e-4 \
    --thr-lumical 4.0e-5 --thr-beamcal 5.0e-5 \
    --thr-muon-barrel 5.0e-6 --thr-muon-endcap 5.0e-6 \
    --collider "${scenario/-/ }" \
    --detector SiD_o2_v04 \
    --legend-fs 12 --logy \
    --hpp-file "${hpp}" \
    --hpp-mu $(python - <<'EOF'
scenario = "${scenario}"
print({
    "C3_250_PS1": 0.059,
    "C3_250_PS2": 0.065,
    "C3_550_PS1": 0.26,
    "C3_550_PS2": 0.29,
}.get(scenario, 0.05))
EOF
    )
done

# Multi-BX production set with scenario-dependent bunch spacing
while read -r scenario spacing maxfiles ratecap; do
  base="/data/GuineaPig_runs/${scenario}/ddsim"
  hpp="/data/hpp/${scenario}/gg_had_MERGED.edm4hep.root"
  python timing_studies/timing_all_BX_Aug2025.py \
    --base-dir "${base}" \
    --pattern "ddsim_${scenario}_v2_seed_*" \
    --bunch-spacing "${spacing}" \
    --max-files "${maxfiles}" \
    --hpp-file "${hpp}" \
    --hpp-mu "${ratecap}" \
    --time-min 0 --time-max 720 \
    --base-binwidth 0.25 \
    --plot-binwidth 10 \
    --out "timing_all_BX_${scenario}.npz" \
    --outdir outputs/all_BX_${scenario} \
    --thr-vertex-barrel 6.0e-7 --thr-vertex-endcap 5.0e-7 \
    --thr-tracker-barrel 3.0e-5 --thr-tracker-endcap 3.0e-5 --thr-tracker-forward 4.0e-7 \
    --thr-ecal-barrel 5.0e-5 --thr-ecal-endcap 5.0e-5 \
    --thr-hcal-barrel 2.4e-4 --thr-hcal-endcap 2.3e-4 \
    --thr-lumical 4.0e-5 --thr-beamcal 5.0e-5 \
    --thr-muon-barrel 5.0e-6 --thr-muon-endcap 5.0e-6 \
    --collider "${scenario/-/ }" \
    --detector SiD_o2_v04 \
    --legend-fs 12 \
    --vb-layers --vb-ymax 200 \
    --rates-json outputs/all_BX_${scenario}/hit_rates.json
done <<'SCENARIOS'
C3_250_BL      5.26 133 0.059
C3_250_SU      2.63 266 0.059
C3_250_highL   2.63 532 0.065
C3_550_BL      3.50  75 0.29
C3_550_SU      1.75 150 0.29
C3_550_highL   1.75 300 0.29
SCENARIOS

```

---

## Implementation Details

- File discovery uses regular expressions to extract numeric seeds and sort input files deterministically.
- Rebinning requires `--plot-binwidth` to be an integer multiple of `--base-binwidth`; otherwise a `ValueError` is raised.
- HPP files are processed in chunks (`step` parameter within the scripts) to manage memory for large event samples.
- Tracker energy thresholds rely on `<branch>.eDep` when present; calorimeter thresholds apply to the first contribution energy (`<branch>Contributions.energy[begin]`).
- The SiVertexBarrel layer logic uses pre-defined layer radii (`VB_LOWER_R`, `VB_UPPER_R`, `VB_RADIAL_CENTERS`) and active length (126 mm) to compute areas.
- Constant fits default to SciPy’s `curve_fit`; if SciPy is unavailable the scripts fall back to sample means and standard errors.
- JSON rate summaries merge with existing files using a nested-dictionary update so previous collider scenarios are preserved.

---

## Acknowledgements

- ROOT I/O and columnar manipulation through `uproot` and `awkward`.
- Numerical routines via `numpy` (and `scipy` when available).
- Plotting provided by `matplotlib`; CMS-inspired styling powered by `mplhep`.


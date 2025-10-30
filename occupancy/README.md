# DD4hep Hit Analysis Framework

A comprehensive framework for analyzing hit-level data from DD4hep detector simulations stored in edm4hep.root format. The framework provides occupancy analysis, timing distributions, hit counting, and geometry parsing capabilities for the SiD detector design.

## Overview

This framework is designed to analyze background hits from particle collider simulations, with particular focus on:

- **Occupancy Analysis**: Calculate cell occupancy as a function of buffer depth (hit count threshold)
- **Timing Distributions**: Analyze hit time distributions to study prompt vs. backscattered hits
- **Geometry Parsing**: Extract detector geometry from SiD XML files (k4geo format)
- **Train-based Analysis**: Process multiple beam trains with statistical averaging
- **Performance Optimizations**: Parallel I/O, parallel train processing, and vectorized hit processing

### Supported Detectors

The framework supports all 13 detector types in the SiD design:

- **Vertex Detectors**: SiVertexBarrel, SiVertexEndcap
- **Tracker System**: SiTrackerBarrel, SiTrackerEndcap, SiTrackerForward  
- **Calorimeters**: ECalBarrel, ECalEndcap, HCalBarrel, HCalEndcap
- **Muon System**: MuonBarrel, MuonEndcap
- **Forward Calorimeters**: BeamCal, LumiCal

## Installation

### Prerequisites

- Python 3.x
- Required packages: `uproot`, `numpy`, `awkward`, `matplotlib`, `mplhep`, `hist`

### Installation Steps

```bash
cd dd4hep_hit_analysis_framework
pip install -e .
# or
python setup.py install
```

### XML File Configuration

The framework requires access to SiD detector XML files. The paths are configured in `src/detector_config.py`:

```python
from src.detector_config import get_xmls
xmls = get_xmls()
# Returns dictionary with paths to main XML and detector-specific XMLs
```

Update the paths in `get_xmls()` if your XML files are located elsewhere.

## Architecture & Components

### Core Modules

#### 1. Detector Configuration (`src/detector_config.py`)

Manages detector-specific settings including cell sizes and detector types:

- **`DetectorConfig` class**: Stores detector name, type (barrel/endcap/forward), class (vertex/tracker/ecal/etc.), and cell sizes
- **`get_detector_configs()`**: Returns dictionary of all 13 detector configurations
- **`get_xmls()`**: Returns dictionary of XML file paths

Key features:
- Default cell sizes per detector class (configurable)
- Layer-specific and region-specific cell size overrides
- Detector type classification for routing to appropriate parsers

#### 2. Geometry Parsing (`src/geometry_parsing/`)

Parses SiD detector XML files to extract physical dimensions and cell counts:

- **`k4geo_parsers.py`**: Main parsing logic
  - `parse_barrel_geometry()`: Parses barrel-type detectors
  - `parse_endcap_geometry()`: Parses endcap-type detectors  
  - `parse_forward_geometry()`: Parses forward calorimeters
  - `parse_calorimeter_geometry()`: Parses ECAL/HCAL detectors
  - `parse_muon_geometry()`: Parses muon system detectors
  - `parse_value()`: Enhanced expression parser with unit conversion
  - `evaluate_math_expression()`: Evaluates arithmetic expressions with operator precedence
  - `derive_tracker_envelope_bounds()`: Fallback derivation of missing geometry bounds

- **`geometry_info.py`**: Unified interface (`get_geometry_info()`) that routes to appropriate parser based on detector class

- **`cellid_decoders.py`**: Decodes DD4hep cellID bitfields into detector-specific fields (layer, module, sensor, etc.)

**Key Capabilities**:
- Handles complex XML expressions: `"SiTrackerBarrel_inner_rc - 6.2*mm"`
- Supports parenthesized expressions: `"(787.105+1.75)*mm"`
- Constant substitution with word-boundary matching
- Fallback derivation when direct parsing fails
- Extracts: module dimensions, layer geometry, cell counts, repeat structures, envelope bounds

#### 3. Hit Analysis (`src/hit_analysis/`)

Core hit processing and analysis functionality:

- **`occupancy.py`**: 
  - `analyze_detector_hits()`: Main hit analysis function with energy threshold filtering
  - `analyze_detector_hits_vectorized()`: Vectorized version for performance
  - `_resolve_energy_thresholds()`: Energy threshold resolution logic
  - `calculate_layer_occupancy()`: Occupancy calculation per layer

- **`train_analyzer.py`**:
  - `analyze_detectors_and_plot_by_train()`: Train-based analysis workflow
  - `average_train_results()`: Statistical averaging across trains
  - `split_seeds_into_trains()`: Organize seeds into train bunches
  - Train-averaged plotting functions

- **`plotting.py`**: Visualization utilities for occupancy and timing plots

#### 4. Segmentation (`src/segmentation/`)

Hit-to-pixel coordinate transformation:

- **`pixelizers.py`**:
  - `get_pixel_id()`: Unified pixel ID calculation
  - `get_tracker_pixel_id()`: Tracker-specific pixelization
  - `get_calo_muon_pixel_id()`: Calorimeter/muon pixelization
  - `get_forward_calo_pixel_id()`: Forward calorimeter pixelization
  - `calculate_local_coordinates()`: Transform global to module-local coordinates
  - `get_pixel_indices()`: Convert local coordinates to pixel indices

#### 5. Performance Utilities (`src/utils/`)

Optimization modules for parallel processing:

- **`parallel_io.py`**: 
  - `open_files_parallel()`: Concurrent ROOT file opening using ThreadPoolExecutor
  - `open_files_in_chunks()`: Memory-safe chunked file opening
  - `validate_file_paths()`: Pre-validation of file existence
  - `estimate_memory_usage()`: Memory requirement estimation

- **`parallel_train_processing.py`**:
  - `process_trains_parallel()`: Multiprocessing for independent train analysis
  - `process_single_train_worker()`: Worker function for train processing
  - `benchmark_train_parallelism()`: Performance benchmarking

- **`vectorized_hit_processing.py`**:
  - `process_hits_vectorized()`: Vectorized hit processing pipeline
  - `decode_cellids_vectorized()`: Fast cellID decoding with NumPy
  - Batch processing of cellIDs and coordinates
  - 10-100x speedup over Python loops

- **`parallel_config.py`**:
  - `get_optimal_worker_count()`: Calculates optimal worker count based on system resources
  - Automatic I/O vs CPU-bound workload detection

- **`histogram_utils.py`**: Utilities for geometry filtering and area calculations

- **`detector_area_helper.py`**: Detector area calculation utilities

## Key Features

### Energy Threshold System

The framework implements a sophisticated two-level energy threshold system for different detector types.

#### Silicon Detectors (Vertex/Tracker)

Single energy deposition threshold (`eDep`):
- Direct threshold on hit energy deposition
- Default: 30 keV (configurable)
- Filtered hits: `energy >= silicon_threshold`

#### Calorimeters (ECAL/HCAL/Muon/Forward)

Two-level threshold system:
1. **Contribution-level threshold**: Filters individual energy contributions
   - Default ECAL: 200 keV
   - Default HCAL: 1 MeV
   - Default Muon: 5 MeV
2. **Hit-level threshold**: Filters hits based on cumulative energy
   - Default ECAL: 5 MeV
   - Default HCAL: 20 MeV
   - Default Muon: 50 MeV

**Hit Time Definition** (configurable via `calo_hit_time_def`):
- `0`: Minimum time of all contributions (default)
- `1`: Time when cumulative energy exceeds hit-level threshold

#### Threshold Resolution Order

Thresholds are resolved with the following priority:
1. **Per-detector override**: `'{detector_name}_{quantity}'` (e.g., `'SiTrackerBarrel_silicon'`, `'ECalEndcap_hits'`)
2. **Class-level default**: Based on detector class (silicon/ecal/hcal/muon)

**Example Configuration**:
```python
custom_thresholds = {
    # Global defaults
    'silicon': 30e-3,              # 30 keV for silicon
    'ecal_hits': 5e-3,              # 5 MeV for ECAL hits
    'ecal_contributions': 0.2e-3,   # 200 keV for ECAL contributions
    
    # Per-detector overrides
    'SiTrackerBarrel_silicon': 3e-5,     # 30 keV
    'ECalBarrel_hits': 5e-5,              # 50 keV
    'ECalBarrel_contributions': 1e-20,    # Very small threshold
}
```

**Units**: All thresholds are in GeV (edm4hep standard).

### Cell Size Configuration

Cell sizes determine the granularity of hit pixelization and directly affect occupancy calculations.

#### Default Cell Sizes

Defined in `DetectorConfig.DEFAULT_CELL_SIZES`:

- **Vertex detectors**: 10×10 μm (`{'x': 0.01, 'y': 0.01}`)
- **Tracker detectors**: 20×20 μm (`{'x': 0.02, 'y': 0.02}`)
- **ECAL**: 3×3 mm (`{'x': 3.0, 'y': 3.0}`)
- **HCAL**: 30×30 mm (`{'x': 30.0, 'y': 30.0}`)
- **Muon**: 30×30 mm (`{'x': 30.0, 'y': 30.0}`)
- **Forward calorimeters**: 3.5×3.5 mm (`{'x': 3.5, 'y': 3.5}`)

#### Cell Size Overrides

Cell sizes can be overridden per detector in analysis scripts:

```python
from src.detector_config import get_detector_configs

DETECTOR_CONFIGS = get_detector_configs()

# Override cell sizes
DETECTOR_CONFIGS['SiTrackerBarrel'].cell_sizes['default'] = {'x': 0.025, 'y': 0.100}
DETECTOR_CONFIGS['ECalBarrel'].cell_sizes['default'] = {'x': 0.025, 'y': 0.100}
```

**Impact on Occupancy**:
- Smaller cell sizes → More cells → Potentially lower occupancy
- Larger cell sizes → Fewer cells → Potentially higher occupancy
- Cell sizes should match the actual readout granularity of the detector

### Geometry Parsing

The geometry parsing system extracts detector dimensions, module layouts, and cell counts from SiD XML files.

#### Expression Parsing

The `parse_value()` function handles complex XML expressions:

- **Simple values**: `"5.0*mm"` → 5.0
- **Constant references**: `"SiTrackerBarrel_inner_rc"` → Substitutes constant value
- **Arithmetic**: `"100.114/2*mm"` → 50.057
- **Parenthesized expressions**: `"(787.105+1.75)*mm"` → 788.855
- **Complex expressions**: `"SiTrackerBarrel_inner_rc - 6.2*mm"` → Evaluates both parts

**Expression Evaluation**:
1. Constant substitution (word-boundary matching)
2. Parenthesized expression evaluation before unit normalization
3. Unit conversion (mm, cm, m, rad, deg, mrad)
4. Arithmetic with proper operator precedence

#### Fallback Derivation

For tracker detectors where envelope bounds aren't directly specified in XML, the framework uses `derive_tracker_envelope_bounds()`:

- Derives `inner_r`, `outer_r` from `rc` (center radius) and module dimensions
- Derives `z_length` from `z0`, `nz`, and module spacing
- Uses constants like `SiTracker_module_z_spacing` and `SiTrackerBarrel_rc_dr` when available

#### Envelope Bounds

All detectors now have properly populated envelope bounds:

- **Barrel detectors**: `inner_r`, `outer_r`, `z_length`
- **Endcap detectors**: `inner_r`, `outer_r`, `z_min`, `z_max`, `z_length` (calculated)
- **Forward detectors**: `inner_r`, `outer_r`, `z_min`, `z_max`

These bounds are used by the geometry mask to filter hits within detector acceptance.

### Parallel Processing Architecture

The framework implements three levels of parallel processing for 10-100x overall speedup:

#### 1. Parallel I/O (`src/utils/parallel_io.py`)

**Concurrent File Opening**:
- Opens multiple ROOT files simultaneously using `ThreadPoolExecutor`
- Automatic worker count optimization: `min(32, cpu_count + 4, num_files)`
- Progress tracking with performance metrics

**Features**:
- Error handling: Individual file failures don't stop the process
- Memory management: Chunked processing for large datasets
- File validation: Pre-validates file existence

**Example**:
```python
from src.utils.parallel_io import open_files_parallel

trees, failed_seeds = open_files_parallel(
    base_path="/path/to/files/",
    filename_pattern="file_{}.root",
    seeds=[1, 2, 3, ..., 1000],
    max_workers=16
)
```

#### 2. Parallel Train Processing (`src/utils/parallel_train_processing.py`)

**Multiprocessing for Trains**:
- Processes multiple trains simultaneously using `multiprocessing.Pool`
- Each train is processed independently with result aggregation
- Automatic error handling and recovery

**Train Organization**:
- Seeds are grouped into trains (e.g., 133 bunches per train)
- Each train represents one beam crossing pattern
- Results are averaged across trains with statistical errors

**Configuration**:
- `TRAIN_BATCH_SIZE`: Number of train batches to process
- `MAX_TRAIN_WORKERS`: Maximum workers per train batch (default: 30)

**Example**:
```python
from src.hit_analysis.train_analyzer import analyze_detectors_and_plot_by_train

results = analyze_detectors_and_plot_by_train(
    detectors_to_analyze=[('SiTrackerBarrel', tracker_xml)],
    all_seeds=seeds,
    bunches_per_train=133,
    train_batch_size=4,
    max_train_workers=30,
    ...
)
```

#### 3. Vectorized Hit Processing (`src/utils/vectorized_hit_processing.py`)

**NumPy Vectorization**:
- Replaces Python loops with NumPy vectorized operations
- Batch processing of cellIDs and coordinates
- 10-100x speedup over traditional Python loops

**Features**:
- Vectorized cellID decoding
- Batch pixel coordinate calculation
- Efficient hit counting per pixel

**Usage**:
Enabled by default via `USE_VECTORIZED_PROCESSING=True` in analysis scripts.

### Endcap / Forward Detector Occupancy

Endcap trackers and forward detectors are mirrored in DD4hep via `reflect="true"`, so each layer exists at +z and -z.  
The framework now tracks the two halves independently and reports both components:

- Pixel keys include the reconstructed side (0 → +z, 1 → −z) derived from the `side` bit and the hit `z` coordinate.
- Per-layer statistics store `occupancy_plus_z`, `occupancy_minus_z`, hit totals, and counts-above-threshold for each side.
- The default `occupancy` field used in plots and summaries is the **average** of the two sides.  
  (This matches the single-disk geometry cell count while exposing the individual sides for cross-checks.)
- Summary rows list both the nominal cell total and an `effective` total (×2 when both sides are present) so per-side and combined averages are easy to verify.
- The terminal summary additionally prints the sensitive area (cm²) obtained from the DD4hep geometry helper for quick cross-checks.
- Verification plots with suffix `_VERIFY_plus_minus_z` overlay the +z (solid) and −z (dashed) curves to highlight asymmetries.

Use the new `interactive_plots` argument in `analyze_detectors_and_plot_by_train()` (and the plotting helpers) to control whether `plt.show()` is called.  
This defaults to `False` for headless batch runs; set `interactive_plots=True` if you want live Matplotlib windows.

## Usage Examples

### Basic Analysis Script

The main analysis script is `analysis_scripts/occupancy_analysis.py`. Key configuration:

```python
from src.detector_config import get_detector_configs, get_xmls
from src.hit_analysis.train_analyzer import analyze_detectors_and_plot_by_train

# Get detector configurations
DETECTOR_CONFIGS = get_detector_configs()
xmls = get_xmls()

# Configure analysis
detectors_to_analyze = [
    ('SiTrackerBarrel', xmls['tracker_barrel_xml']),
    ('SiTrackerEndcap', xmls['tracker_endcap_xml']),
]

# Define seeds (one per bunch crossing)
seeds = [106733, 107626, 112380, ...]  # Your seed list

# Configure energy thresholds
custom_thresholds = {
    'silicon': 0.6e-6,                    # 0.6 keV default
    'SiTrackerBarrel_silicon': 3e-5,      # 30 keV override
    'ecal_hits': 50e-6,
    'hcal_hits': 0.24e-3,
}

# Override cell sizes
DETECTOR_CONFIGS['SiTrackerBarrel'].cell_sizes['default'] = {'x': 0.025, 'y': 0.100}

# Run analysis
results = analyze_detectors_and_plot_by_train(
    DETECTOR_CONFIGS=DETECTOR_CONFIGS,
    detectors_to_analyze=detectors_to_analyze,
    all_seeds=seeds,
    bunches_per_train=133,
    main_xml=xmls['main_xml'],
    base_path="/path/to/root/files/",
    filename_pattern="ddsim_C3_250_PS1_v2_seed_{}.edm4hep.root",
    remove_zeros=True,
    time_cut=-1,                          # No time cut
    calo_hit_time_def=1,                  # Cumulative energy for calorimeters
    energy_thresholds=custom_thresholds,
    use_vectorized_processing=True,
    train_batch_size=4,
    max_train_workers=30,
)
```

### Custom Configuration

#### Energy Thresholds

```python
# Global defaults
energy_thresholds = {
    'silicon': 30e-3,              # 30 keV
    'ecal_hits': 5e-3,             # 5 MeV
    'ecal_contributions': 0.2e-3,   # 200 keV
    'hcal_hits': 20e-3,            # 20 MeV
    'hcal_contributions': 1e-3,     # 1 MeV
}

# Per-detector overrides
energy_thresholds['SiTrackerBarrel_silicon'] = 3e-5  # 30 keV
energy_thresholds['ECalBarrel_hits'] = 5e-5          # 50 keV
```

#### Cell Size Overrides

```python
# Override per detector
DETECTOR_CONFIGS['SiTrackerBarrel'].cell_sizes['default'] = {'x': 0.025, 'y': 0.100}
DETECTOR_CONFIGS['ECalBarrel'].cell_sizes['default'] = {'x': 0.025, 'y': 0.100}

# Layer-specific overrides (if needed)
DETECTOR_CONFIGS['SiTrackerBarrel'].cell_sizes[1] = {'x': 0.03, 'y': 0.12}
```

#### Buffer Depths

Buffer depths define the hit count thresholds for occupancy calculation:

```python
buffer_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Occupancy calculated for cells with hits >= each threshold
```

#### Time Cuts

Apply time cuts to filter hits:

```python
time_cut = 5.25 * 133  # 5.25 ns × bunches_per_train in ns
# Only hits with time < time_cut are included
```

#### Geometry Filter Tolerances

Tolerance for geometry filtering (hits outside detector bounds):

```python
GEOMETRY_FILTER_TOLERANCES = {
    'default': 1.0,                    # 1 mm default
    'SiVertexBarrel': 0.9,             # 0.9 mm for vertex
    'SiTrackerEndcap': 0.9,
}
```

## Workflow & Data Flow

The analysis workflow follows these steps:

### 1. Geometry Parsing

```
XML Files → parse_detector_constants() → Constants Dictionary
         → get_geometry_info() → Geometry Info Dictionary
                               → Layer dimensions, module layouts
                               → Cell counts per layer
                               → Envelope bounds (inner_r, outer_r, z_length)
```

**Output**: `geometry_info` dictionary with per-layer geometry and total cell counts

### 2. Hit Reading

```
ROOT Files → open_files_parallel() → Event Trees
          → Read hit collections (cellID, position, energy, time)
          → Apply energy thresholds
          → Apply time cuts
          → Filter zero positions
```

**Branches Read**:
- Silicon: `cellID`, `position.x/y/z`, `eDep`, `time`
- Calorimeters: `cellID`, `position.x/y/z`, `energy`, `contributions` (time, energy)

### 3. Hit Processing

```
Hits → decode_dd4hep_cellid() → Decoded fields (layer, module, sensor, ...)
     → get_pixel_id() → Pixel coordinates (layer, module, pixel_x, pixel_y)
     → Count hits per pixel → Pixel hit dictionary
```

**Pixel Key Structure**:
- Tracker barrel: `(layer, module, pixel_t, pixel_z)`
- Tracker endcap: `(layer, module, pixel_x, pixel_y)` (sums +z/-z sides)
- Calorimeters: `(layer, module, stave, slice, pixel_x, pixel_y)`
- Forward calos: `(layer, slice, cell_x, cell_y)`

### 4. Occupancy Calculation

```
Pixel Hits → For each buffer depth threshold:
            → Count pixels with hits >= threshold
            → Calculate: occupancy = cells_above_threshold / total_cells
            → Per-layer statistics
```

**Statistics Calculated**:
- `cells_hit`: Number of cells with at least one hit
- `cells_above_threshold`: Number of cells with hits >= threshold
- `total_hits`: Total hit count
- `occupancy`: Fraction of cells above threshold
- `mean_hits`: Average hits per cell

### 5. Train Averaging

```
Individual Train Results → average_train_results() → Averaged Statistics
                        → Calculate mean, std_dev, std_error per layer
                        → Aggregate across trains
```

**Output**: Train-averaged occupancy with statistical errors

### 6. Visualization

```
Averaged Statistics → plot_train_averaged_occupancy_analysis() → Occupancy Plots
                   → plot_train_averaged_timing_analysis() → Timing Density Plots
```

**Plot Types**:
- Occupancy vs buffer depth (per layer, batched)
- Occupancy vs buffer (buffer-only comparison)
- Timing density distributions (r-phi and r-z views)

## File Structure

```
dd4hep_hit_analysis_framework/
├── README.md                          # This file
├── setup.py                           # Package installation
├── analysis_scripts/
│   ├── occupancy_analysis.py         # Main analysis script
│   ├── simple_occupancy_analysis.py  # Simple analysis example
│   └── simple_plotting.py            # Basic plotting example
├── src/
│   ├── detector_config.py            # Detector configuration
│   ├── geometry_parsing/
│   │   ├── k4geo_parsers.py         # Main XML parsing logic
│   │   ├── geometry_info.py         # Unified geometry interface
│   │   ├── cellid_decoders.py       # CellID bitfield decoding
│   │   └── geometry_analyzers.py    # Geometry analysis utilities
│   ├── hit_analysis/
│   │   ├── occupancy.py             # Hit analysis and occupancy calculation
│   │   ├── train_analyzer.py        # Train-based analysis workflow
│   │   ├── plotting.py              # Plotting utilities
│   │   └── read_hits.py             # Hit reading utilities
│   ├── segmentation/
│   │   └── pixelizers.py            # Hit-to-pixel coordinate transformation
│   └── utils/
│       ├── parallel_io.py           # Parallel ROOT file opening
│       ├── parallel_train_processing.py  # Multiprocessing for trains
│       ├── vectorized_hit_processing.py  # Vectorized hit processing
│       ├── parallel_config.py       # Worker count optimization
│       ├── histogram_utils.py      # Geometry filtering utilities
│       └── detector_area_helper.py  # Detector area calculations
├── GEOMETRY_PARSING_FIXES.md        # Documentation of geometry parsing fixes
├── PARALLEL_IO_README.md            # Detailed parallel I/O documentation
└── GEOMETRY_INTERPRETATION_GUIDE.md  # Geometry interpretation guide
```

## Configuration Reference

### Energy Threshold Defaults

| Detector Class | Quantity | Default Value | Units |
|---------------|----------|---------------|-------|
| Silicon | `silicon` | 30e-3 | GeV (30 keV) |
| ECAL | `hits` | 5e-3 | GeV (5 MeV) |
| ECAL | `contributions` | 0.2e-3 | GeV (200 keV) |
| HCAL | `hits` | 20e-3 | GeV (20 MeV) |
| HCAL | `contributions` | 1e-3 | GeV (1 MeV) |
| Muon | `hits` | 50e-3 | GeV (50 MeV) |
| Muon | `contributions` | 5e-3 | GeV (5 MeV) |

### Default Cell Sizes

| Detector Class | Cell Size (x, y) | Units |
|---------------|------------------|-------|
| Vertex | 0.01 × 0.01 | mm |
| Tracker | 0.02 × 0.02 | mm |
| ECAL | 3.0 × 3.0 | mm |
| HCAL | 30.0 × 30.0 | mm |
| Muon | 30.0 × 30.0 | mm |
| Forward Calos | 3.5 × 3.5 | mm |

### Performance Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_BATCH_SIZE` | 4 | Number of train batches for parallel processing |
| `MAX_TRAIN_WORKERS` | 30 | Maximum workers per train batch |
| `USE_VECTORIZED_PROCESSING` | True | Enable vectorized hit processing |
| I/O Worker Count | `min(32, cpu_count + 4, num_files)` | Optimal for parallel file opening |
| CPU Worker Count | `min(cpu_count, num_files)` | Optimal for CPU-bound operations |

## Output & Results

### Generated Files

The analysis produces PDF and PNG plots for each detector:

- **Occupancy plots**: `{detector}_train{bunches}_occupancy.pdf/png`
  - Per-layer occupancy vs buffer depth
  - Multiple layers batched per plot
  - Error bars from train-to-train variation

- **Occupancy vs buffer**: `{detector}_train{bunches}_occupancy_vs_buffer_only.pdf/png`
  - Focused view of buffer depth dependence

- **Timing density plots**: `{detector}_train{bunches}_timing_density.pdf/png`
  - Hit time distributions in r-phi and r-z views
  - Useful for studying prompt vs backscattered hits

### Statistics Output

Console output includes:
- Per-layer occupancy statistics
- Total cells per layer
- Train-averaged values with standard errors
- Overall detector statistics

## Troubleshooting & Best Practices

### Common Issues

#### Geometry Parsing Failures

**Problem**: Missing envelope bounds (`inner_r`, `outer_r`, `z_length` are `None`)

**Solution**: 
- Check that XML files are accessible and properly formatted
- Verify constants are being parsed correctly (`parse_detector_constants()`)
- Check for parsing warnings in console output
- Review `GEOMETRY_PARSING_FIXES.md` for known issues

#### Performance Issues

**Problem**: Slow processing despite parallelization

**Possible Causes**:
- Storage system doesn't support concurrent access (NFS/network storage)
- Too many workers saturating I/O system
- Memory bottlenecks

**Solutions**:
- Reduce worker count for network storage (`max_workers=4-8`)
- Use chunked processing for large datasets
- Monitor system resources (CPU, memory, I/O)

#### Memory Issues

**Problem**: Out of memory errors with large datasets

**Solutions**:
- Use `open_files_in_chunks()` instead of opening all files at once
- Reduce `TRAIN_BATCH_SIZE` or `MAX_TRAIN_WORKERS`
- Process detectors sequentially instead of in parallel
- Increase system memory or use a machine with more RAM

#### Occupancy Values Too High/Low

**Check**:
- Energy thresholds: Lower thresholds → more hits → higher occupancy
- Cell sizes: Smaller cells → more cells → potentially lower occupancy
- For endcap detectors: Occupancy is total across both sides (≈2× per-side)

### Best Practices

1. **Validate File Paths**: Use `validate_file_paths()` before processing large datasets
2. **Estimate Memory**: Use `estimate_memory_usage()` for datasets with 100+ files
3. **Monitor Progress**: Watch console output for file opening progress and errors
4. **Start Small**: Test with a small subset of seeds before full analysis
5. **Check Geometry**: Verify geometry parsing output for correct cell counts
6. **Energy Thresholds**: Start with conservative thresholds and adjust based on results
7. **Train Size**: Use consistent `bunches_per_train` across analyses for comparison

### Performance Tuning

**For Network Storage (NFS)**:
- Reduce `max_workers` to 4-8 for parallel I/O
- Use larger chunk sizes
- Monitor network bandwidth

**For Local SSD Storage**:
- Can use higher worker counts (16-32)
- Smaller chunk sizes acceptable
- Monitor CPU utilization

**For Large Datasets (500+ files)**:
- Use chunked processing
- Process detectors sequentially
- Increase `TRAIN_BATCH_SIZE` to reduce overhead

## Related Documentation

- **`GEOMETRY_PARSING_FIXES.md`**: Detailed documentation of geometry parsing enhancements
- **`PARALLEL_IO_README.md`**: Comprehensive guide to parallel I/O optimizations
- **`GEOMETRY_INTERPRETATION_GUIDE.md`**: Guide to interpreting geometry XML structures

## Quick Start

```bash
# 1. Install framework
cd dd4hep_hit_analysis_framework
pip install -e .

# 2. Configure paths in src/detector_config.py (get_xmls())

# 3. Run analysis
cd analysis_scripts
python occupancy_analysis.py
```

## Contributing

When modifying the framework:

1. **Geometry Parsing**: Ensure new parsers extract envelope bounds (`inner_r`, `outer_r`, `z_length` or `z_min`/`z_max`)
2. **Cell Sizes**: Update defaults in `DetectorConfig.DEFAULT_CELL_SIZES` if needed
3. **Energy Thresholds**: Follow the resolution order: per-detector override → class default
4. **Performance**: Use vectorized operations where possible, leverage parallel processing
5. **Testing**: Test with all 13 detector types to ensure no regressions

## License

See LICENSE file for details.

## Contact

For questions or issues, contact: dntounis@stanford.edu

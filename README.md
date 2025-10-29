# Hadron photoproduction

## Instructions for running Circe/whizard for photon hadronproduction
#### Originally written by Lindsey Gray (FNAL), adjusted for use on SLAC S3DF by Dimitris Ntounis (Stanford/SLAC)

## Repository contents

- `S3DF/` – batch submission workflow for producing hadron photoproduction samples and subsequent detector simulation on the SLAC S3DF Slurm cluster. The wrapper scripts (`S3DF_whizard_for_gg_to_hadrons.sh`, `payload_whizard_for_gg_to_hadrons.sh`, and their DDSim counterparts) stage inputs, set up the required environments, and submit jobs in the expected queue structure. Users targeting a different site should treat these as templates and adapt the queue names, module loads, and storage paths to match their scheduler and filesystem.
- `scripts/` – Python utilities for analysing luminosity spectra and detector outputs. They cover GuineaPig vs. Circe comparisons (`python_compare_ratio.py`, `python_compare.py`), direct plotting from Circe-generated `.gp` histograms (`plot_gp_normalized_from_gpfiles.py`, `plot_gp_normalized_spectra.py`), and visualising downstream particle spectra for hadron photoproduction (HPP) and incoherent pair creation (IPC) (`plot_ipc_vs_hpp_momenta.py`). The scripts assume the directory layout described below; adjust the input arguments if your workspace differs.

1) Prepare the area:

```
#Inside apptainer with ATLAS env set up

mkdir aahadhad
cd aahadhad
mkdir work
# cp the file beams.circe and circe_AA_dd_gg_had_ps_lcio_2025.00.gg.sin from this repo into the work folder
wget https://lcgpackages.web.cern.ch/tarFiles/sources/MCGeneratorsTarFiles/whizard-3.1.5.tar.gz
tar -xzf whizard-3.1.5.tar.gz
cd whizard-3.1.5
```


2) Create environments you'll use:

- make a file `build_env.sh` with the following contents

```
#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt/setup.sh
export F77=$FC
export TIRPC_CFLAGS=-I/usr/include/tirpc
export TIRPC_LIBS="-L/usr/lib64 -ltirpc"
```

- make a file `runtime_env.sh` with the following contents


```
#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt/setup.sh
export F77=$FC
export PATH=${PWD}/whizard-3.1.5-install/bin:$PATH
export LD_LIBRARY_PATH=${PWD}/whizard-3.1.5-install/lib:$LD_LIBRARY_PATH
```

3) Build Whizard:


```
source build_env.sh
mkdir build 
cd build
../configure --prefix ${PWD}/../whizard-3.1.5-install
make -j 9
make install
```

4) Log out of your machine and log back in (or otherwise completely reset the shell env)

5) Get the runtime env and test your install


```
cd aahadhad/whizard-3.1.5/
source runtime_env.sh
cd ../work
echo "process ee = e1, E1 => e2, E2
sqrts = 360 GeV
n_events = 10
sample_format = lhef
simulate (ee)" > hello_world.sin
whizard < hello_world.sin
```

If this completes without issue, you're good to go

6) Trial circe run using the attached file

```
whizard circe_AA_dd_gg_had_ps_lcio_2025.00.gg.sin

```

If this works you're good to go

7) Prepare `beams.circe` file

The process for obtaining the circe file is given in `aahadhad/whizard-3.1.5/circe2/share/examples`

Here are the relevant steps for running this process on S3DF (adjust paths accordingly):

```
#work dir:
WORK=/fs/ddn/sdf/group/atlas/d/dntounis/C^3/hadron_photoproduction/aahadhad/work

#for multiple runs, its recommended to set up a clean work dir for each one, e.g.
WORK_C3_250_PS1=/fs/ddn/sdf/group/atlas/d/dntounis/C^3/hadron_photoproduction/aahadhad/work_C3_250_PS1
WORK_C3_250_PS2=/fs/ddn/sdf/group/atlas/d/dntounis/C^3/hadron_photoproduction/aahadhad/work_C3_250_PS2
WORK_C3_550_PS1=/fs/ddn/sdf/group/atlas/d/dntounis/C^3/hadron_photoproduction/aahadhad/work_C3_550_PS1
WORK_C3_550_PS2=/fs/ddn/sdf/group/atlas/d/dntounis/C^3/hadron_photoproduction/aahadhad/work_C3_550_PS2


# path where the GuineaPig lumi files are stored:
GP_OUTPUT_PATH_C3_250_PS1=/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/output_new/C3_250_PS1
GP_OUTPUT_PATH_C3_250_PS2=/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/output_new/C3_250_PS2
GP_OUTPUT_PATH_C3_550_PS1=/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/output_new/C3_550_PS1
GP_OUTPUT_PATH_C3_550_PS2=/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/output_new/C3_550_PS2


# Concatenate all runs with different seeds to produce single, high-stats files:
cat "$GP_OUTPUT_PATH_C3_250_PS1"/lumi/lumi_ee_C3_250_PS1_seed_*.out > "$WORK_C3_250_PS1/C3_250_PS1_lumi.ee.out"
cat "$GP_OUTPUT_PATH_C3_250_PS1"/lumi/lumi_eg_C3_250_PS1_seed_*.out > "$WORK_C3_250_PS1/C3_250_PS1_lumi.eg.out"
cat "$GP_OUTPUT_PATH_C3_250_PS1"/lumi/lumi_ge_C3_250_PS1_seed_*.out > "$WORK_C3_250_PS1/C3_250_PS1_lumi.ge.out"
cat "$GP_OUTPUT_PATH_C3_250_PS1"/lumi/lumi_gg_C3_250_PS1_seed_*.out > "$WORK_C3_250_PS1/C3_250_PS1_lumi.gg.out"

cat "$GP_OUTPUT_PATH_C3_250_PS2"/lumi/lumi_ee_C3_250_PS2_seed_*.out > "$WORK_C3_250_PS2/C3_250_PS2_lumi.ee.out"
cat "$GP_OUTPUT_PATH_C3_250_PS2"/lumi/lumi_eg_C3_250_PS2_seed_*.out > "$WORK_C3_250_PS2/C3_250_PS2_lumi.eg.out"
cat "$GP_OUTPUT_PATH_C3_250_PS2"/lumi/lumi_ge_C3_250_PS2_seed_*.out > "$WORK_C3_250_PS2/C3_250_PS2_lumi.ge.out"
cat "$GP_OUTPUT_PATH_C3_250_PS2"/lumi/lumi_gg_C3_250_PS2_seed_*.out > "$WORK_C3_250_PS2/C3_250_PS2_lumi.gg.out"

cat "$GP_OUTPUT_PATH_C3_550_PS1"/lumi/lumi_ee_C3_550_PS1_seed_*.out > "$WORK_C3_550_PS1/C3_550_PS1_lumi.ee.out"
cat "$GP_OUTPUT_PATH_C3_550_PS1"/lumi/lumi_eg_C3_550_PS1_seed_*.out > "$WORK_C3_550_PS1/C3_550_PS1_lumi.eg.out"
cat "$GP_OUTPUT_PATH_C3_550_PS1"/lumi/lumi_ge_C3_550_PS1_seed_*.out > "$WORK_C3_550_PS1/C3_550_PS1_lumi.ge.out"
cat "$GP_OUTPUT_PATH_C3_550_PS1"/lumi/lumi_gg_C3_550_PS1_seed_*.out > "$WORK_C3_550_PS1/C3_550_PS1_lumi.gg.out"

cat "$GP_OUTPUT_PATH_C3_550_PS2"/lumi/lumi_ee_C3_550_PS2_seed_*.out > "$WORK_C3_550_PS2/C3_550_PS2_lumi.ee.out"
cat "$GP_OUTPUT_PATH_C3_550_PS2"/lumi/lumi_eg_C3_550_PS2_seed_*.out > "$WORK_C3_550_PS2/C3_550_PS2_lumi.eg.out"
cat "$GP_OUTPUT_PATH_C3_550_PS2"/lumi/lumi_ge_C3_550_PS2_seed_*.out > "$WORK_C3_550_PS2/C3_550_PS2_lumi.ge.out"
cat "$GP_OUTPUT_PATH_C3_550_PS2"/lumi/lumi_gg_C3_550_PS2_seed_*.out > "$WORK_C3_550_PS2/C3_550_PS2_lumi.gg.out"

# Produce Circe2 input files:
EX=/fs/ddn/sdf/group/atlas/d/dntounis/C^3/hadron_photoproduction/aahadhad/whizard-3.1.5/circe2/share/examples
# one file containing lines like lumi_ee, f_rep, n_b, this is to calculate the instantaneous lumi
GP_SUMMARY_C3_250_PS1=/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/output_new/C3_250_PS1/test_C3_250_PS1_seed_469163.ref
GP_SUMMARY_C3_250_PS2=/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/output_new/C3_250_PS2/test_C3_250_PS2_seed_469163.ref
GP_SUMMARY_C3_550_PS1=/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/output_new/C3_550_PS1/test_C3_550_PS1_seed_469163.ref
GP_SUMMARY_C3_550_PS2=/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/output_new/C3_550_PS2/test_C3_550_PS2_seed_469163.ref

   
PREFIX="$WORK"                               # keep trailing slash so outputs go into WORK, whatever is added after as prefix is prepended to beams.circe and the lumi files!

#Or, for multiple dirs:
PREFIX_C3_250_PS1="$WORK_C3_250_PS1"
PREFIX_C3_250_PS2="$WORK_C3_250_PS2"
PREFIX_C3_550_PS1="$WORK_C3_550_PS1"
PREFIX_C3_550_PS2="$WORK_C3_550_PS2"


DESIGN=C3                                    # or any tag you prefer

sh "$EX/fill_circe2_template" "$GP_SUMMARY_C3_250_PS1" "$EX/template.circe2_input" prefix="$PREFIX_C3_250_PS1/C3_250_PS1_" design="$DESIGN" roots=250 > "$WORK_C3_250_PS1/circe2_input_C3_250_PS1"
sh "$EX/fill_circe2_template" "$GP_SUMMARY_C3_250_PS2" "$EX/template.circe2_input" prefix="$PREFIX_C3_250_PS2/C3_250_PS2_" design="$DESIGN" roots=250 > "$WORK_C3_250_PS2/circe2_input_C3_250_PS2"
sh "$EX/fill_circe2_template" "$GP_SUMMARY_C3_550_PS1" "$EX/template.circe2_input" prefix="$PREFIX_C3_550_PS1/C3_550_PS1_" design="$DESIGN" roots=550 > "$WORK_C3_550_PS1/circe2_input_C3_550_PS1"
sh "$EX/fill_circe2_template" "$GP_SUMMARY_C3_550_PS2" "$EX/template.circe2_input" prefix="$PREFIX_C3_550_PS2/C3_550_PS2_" design="$DESIGN" roots=550 > "$WORK_C3_550_PS2/circe2_input_C3_550_PS2"
```

Before proceeding, open the input circe files (e.g. "$WORK_C3_250_PS1/circe2_input_C3_250_PS1") and make the following changes, this helps improve the circe interpolation and bring the interpolated spectra closer to the GuineaPig ones:

```
#Change number of bins
bins = 120

#ee lumi - add below min = 0 max = 1 fix = *

#for 250

map = null { 1 [0, 0.3] }
map = power { 119 [0.3, 1] beta = -0.7 eta = 1 }

#for 550

map = null { 1 [0, 0.2] }
map = power { 119 [0.2, 1] beta = -0.7 eta = 1 }

# gg lumi - add below min = 0 max = 1 fix = *

#for 250

map = null { 1 [0.4, 1] }
map = power { 119 [0, 0.4] beta = -0.7 eta = 0 }

#for 550

map = null { 1 [0.6, 1] }
map = power { 119 [0, 0.6] beta = -0.7 eta = 0 }
      
```

Now you can proceed with running circe: 

```
# main step: Run circe2!
# Important: if you want to run circe2_compare below, you can skip this step to save time since circe2_compare reruns circe2_tool.opt!


circe2_tool.opt -f "$WORK_C3_250_PS1/circe2_input_C3_250_PS1" > "$WORK_C3_250_PS1/circe2_C3_250_PS1.log" 2>&1
circe2_tool.opt -f "$WORK_C3_250_PS2/circe2_input_C3_250_PS2" > "$WORK_C3_250_PS2/circe2_C3_250_PS2.log" 2>&1
circe2_tool.opt -f "$WORK_C3_550_PS1/circe2_input_C3_550_PS1" > "$WORK_C3_550_PS1/circe2_C3_550_PS1.log" 2>&1
circe2_tool.opt -f "$WORK_C3_550_PS2/circe2_input_C3_550_PS2" > "$WORK_C3_550_PS2/circe2_C3_550_PS2.log" 2>&1

# Compare spectra between circe and GuineaPig

#First, create symlinks, since the circe2_compare script expects files of the form lumi.$gp_tag.out:
for ch in ee eg ge gg; do
  ln -sf "$WORK_C3_250_PS1/C3_250_PS1_lumi.$ch.out" "$WORK_C3_250_PS1/lumi.$ch.out"
done

for ch in ee eg ge gg; do
  ln -sf "$WORK_C3_250_PS2/C3_250_PS2_lumi.$ch.out" "$WORK_C3_250_PS2/lumi.$ch.out"
done

for ch in ee eg ge gg; do
  ln -sf "$WORK_C3_550_PS1/C3_550_PS1_lumi.$ch.out" "$WORK_C3_550_PS1/lumi.$ch.out"
done

for ch in ee eg ge gg; do
  ln -sf "$WORK_C3_550_PS2/C3_550_PS2_lumi.$ch.out" "$WORK_C3_550_PS2/lumi.$ch.out"
done


#Last two arguments:
$5 (2000000): Optional num_events for circe2_generate (default 10,000,000). Higher = smoother Circe2 curves, slower.
$6 (1000): Optional num_bins for histograms (default 1000). Higher = finer bins, can be noisy if events are low.
If you omit $5/$6, it uses 10,000,000 events and 1000 bins.

OUT_PREFIX=C3_250_PS1 sh "$EX/circe2_compare" "$WORK_C3_250_PS1" "$WORK_C3_250_PS1/circe2_input_C3_250_PS1" "$DESIGN" 250 10000000 1000
OUT_PREFIX=C3_250_PS2 sh "$EX/circe2_compare" "$WORK_C3_250_PS2" "$WORK_C3_250_PS2/circe2_input_C3_250_PS2" "$DESIGN" 250 10000000 1000
OUT_PREFIX=C3_550_PS1 sh "$EX/circe2_compare" "$WORK_C3_550_PS1" "$WORK_C3_550_PS1/circe2_input_C3_550_PS1" "$DESIGN" 550 10000000 1000
OUT_PREFIX=C3_550_PS2 sh "$EX/circe2_compare" "$WORK_C3_550_PS2" "$WORK_C3_550_PS2/circe2_input_C3_550_PS2" "$DESIGN" 550 10000000 1000
```

*Note: If the compare script fails, it could be because the pdf terminal is missing from gnuplot. In that case you can try e.g. with svg:

```
cp "$EX/circe2_compare" "$WORK/circe2_compare_svg"
```


```
cp "$EX/circe2_compare" "$WORK_C3_250_PS1/circe2_compare_svg"
cp "$EX/circe2_compare" "$WORK_C3_250_PS2/circe2_compare_svg"
cp "$EX/circe2_compare" "$WORK_C3_550_PS1/circe2_compare_svg"
cp "$EX/circe2_compare" "$WORK_C3_550_PS2/circe2_compare_svg"

```

```
# change PDF terminal and extension to SVG
sed -i 's/set term pdf;/set term svg;/' "$WORK/circe2_compare_svg"
sed -i 's/\.pdf"/.svg"/' "$WORK/circe2_compare_svg"
```

```
# change PDF terminal and extension to SVG
sed -i 's/set term pdf;/set term svg;/' "$WORK_C3_250_PS1/circe2_compare_svg"
sed -i 's/\.pdf"/.svg"/' "$WORK_C3_250_PS1/circe2_compare_svg"

sed -i 's/set term pdf;/set term svg;/' "$WORK_C3_250_PS2/circe2_compare_svg"
sed -i 's/\.pdf"/.svg"/' "$WORK_C3_250_PS2/circe2_compare_svg"

sed -i 's/set term pdf;/set term svg;/' "$WORK_C3_550_PS1/circe2_compare_svg"
sed -i 's/\.pdf"/.svg"/' "$WORK_C3_550_PS1/circe2_compare_svg"

sed -i 's/set term pdf;/set term svg;/' "$WORK_C3_550_PS2/circe2_compare_svg"
sed -i 's/\.pdf"/.svg"/' "$WORK_C3_550_PS2/circe2_compare_svg"
```


and rerun:

```
OUT_PREFIX=C3_250_PS1 sh "$WORK_C3_250_PS1/circe2_compare_svg" "$WORK_C3_250_PS1" "$WORK_C3_250_PS1/circe2_input_C3_250_PS1" "$DESIGN" 250 2000000 1000
OUT_PREFIX=C3_250_PS2 sh "$WORK_C3_250_PS2/circe2_compare_svg" "$WORK_C3_250_PS2" "$WORK_C3_250_PS2/circe2_input_C3_250_PS2" "$DESIGN" 250 2000000 1000
OUT_PREFIX=C3_550_PS1 sh "$WORK_C3_550_PS1/circe2_compare_svg" "$WORK_C3_550_PS1" "$WORK_C3_550_PS1/circe2_input_C3_550_PS1" "$DESIGN" 550 2000000 1000
OUT_PREFIX=C3_550_PS2 sh "$WORK_C3_550_PS2/circe2_compare_svg" "$WORK_C3_550_PS2" "$WORK_C3_550_PS2/circe2_input_C3_550_PS2" "$DESIGN" 550 2000000 1000
```

To get comparison plots:

```
python scripts/python_compare_ratio.py --dir "$WORK_C3_250_PS1" --out-prefix C3_250_PS1 --format pdf --ratio-ylim 0.5 1.5
python scripts/python_compare_ratio.py --dir "$WORK_C3_250_PS2" --out-prefix C3_250_PS2 --format pdf --ratio-ylim 0.5 1.5 
python scripts/python_compare_ratio.py --dir "$WORK_C3_550_PS1" --out-prefix C3_550_PS1 --format pdf --ratio-ylim 0.5 1.5 
python scripts/python_compare_ratio.py --dir "$WORK_C3_550_PS2" --out-prefix C3_550_PS2 --format pdf --ratio-ylim 0.5 1.5 
```

To get Guinea-Pig only spectra:

```
python scripts/plot_gp_normalized_spectra.py \
  --set C3_250_PS1:250:ee="$WORK_C3_250_PS1/lumi.ee.out",eg="$WORK_C3_250_PS1/lumi.eg.out",ge="$WORK_C3_250_PS1/lumi.ge.out",gg="$WORK_C3_250_PS1/lumi.gg.out" \
  --set C3_250_PS2:250:ee="$WORK_C3_250_PS2/lumi.ee.out",eg="$WORK_C3_250_PS2/lumi.eg.out",ge="$WORK_C3_250_PS2/lumi.ge.out",gg="$WORK_C3_250_PS2/lumi.gg.out" \
  --set C3_550_PS1:550:ee="$WORK_C3_550_PS1/lumi.ee.out",eg="$WORK_C3_550_PS1/lumi.eg.out",ge="$WORK_C3_550_PS1/lumi.ge.out",gg="$WORK_C3_550_PS1/lumi.gg.out" \
  --set C3_550_PS2:550:ee="$WORK_C3_550_PS2/lumi.ee.out",eg="$WORK_C3_550_PS2/lumi.eg.out",ge="$WORK_C3_550_PS2/lumi.ge.out",gg="$WORK_C3_550_PS2/lumi.gg.out" \
  --bins 200 --range 0 1 --out "gp_norm"
```

or, if you've already run circe before to get the .gp files you can save time by using this script: 

```
python scripts/plot_gp_normalized_from_gpfiles.py \
  --set C3_250_PS1:250:dir=$WORK_C3_250_PS1 \
  --set C3_250_PS2:250:dir=$WORK_C3_250_PS2 \
  --set C3_550_PS1:550:dir=$WORK_C3_550_PS1 \
  --set C3_550_PS2:550:dir=$WORK_C3_550_PS2 \
  --bins-ee 2000 --bins-eg 400 --bins-ge 400 --bins-gg 400 \
  --out "gp_norm_from_gp" --format pdf --logy --ymin 1e-4
```

To visualise hadron photoproduction (HPP) spectra alongside the incoherent pair creation (IPC) backgrounds, first convert the Whizard `.slcio` samples into EDM4hep (identify missing collections with `check_missing_cols --minimal`, regenerate with `lcio2edm4hep`, and merge per parameter set using `hadd`). With the converted HPP files and the GuineaPig particle ntuples prepared, you can run:

```
python scripts/plot_ipc_vs_hpp_momenta.py \
  --ipc-C3-250-PS1 $GP_OUTPUT_PATH_C3_250_PS1 \
  --ipc-C3-250-PS2 $GP_OUTPUT_PATH_C3_250_PS2 \
  --ipc-C3-550-PS1 $GP_OUTPUT_PATH_C3_550_PS1 \
  --ipc-C3-550-PS2 $GP_OUTPUT_PATH_C3_550_PS2 \
  --hpp-C3-250-PS1 $WORK_C3_250_PS1/circe_AA_dd_ps_nofilter_00_new_MERGED.edm4hep.root \
  --hpp-C3-250-PS2 $WORK_C3_250_PS2/circe_AA_dd_ps_nofilter_00_new_MERGED.edm4hep.root \
  --hpp-C3-550-PS1 $WORK_C3_550_PS1/circe_AA_dd_ps_nofilter_00_new_MERGED.edm4hep.root \
  --hpp-C3-550-PS2 $WORK_C3_550_PS2/circe_AA_dd_ps_nofilter_00_new_MERGED.edm4hep.root \
  --hpp-final-state --ipc-max-files 500 \
  --bins 180 --density --logy \
  --energy-xmin 0 --energy-xmax 200 \
  --pt-xmin 0 --pt-xmax 20 \
  --pz-xmin -200 --pz-xmax 200 \
  --format pdf --workers 64 --log-level INFO \
  --pdg-xmin -10000 --pdg-xmax 10000 \
  --save-npz --load-npz --npz-dir $WORK/particle_momenta_plots_cache \
  --out-dir $WORK/particle_momenta_plots_latest
```

This produces matched histograms for IPC and HPP spectra (energy, transverse momentum, longitudinal momentum, and particle-type) while reusing cached histograms when available.



8) Now run whizard with C3 circe input file

Copy paste the template sin file
```
cp circe_AA_dd_gg_had_ps_lcio_2025.00.gg.sin "$WORK_C3_250_PS1/circe_AA_dd_gg_had_ps_C3_250_PS1.gg.sin"
cp circe_AA_dd_gg_had_ps_lcio_2025.00.gg.sin "$WORK_C3_250_PS2/circe_AA_dd_gg_had_ps_C3_250_PS2.gg.sin"
cp circe_AA_dd_gg_had_ps_lcio_2025.00.gg.sin "$WORK_C3_550_PS1/circe_AA_dd_gg_had_ps_C3_550_PS1.gg.sin"
cp circe_AA_dd_gg_had_ps_lcio_2025.00.gg.sin "$WORK_C3_550_PS2/circe_AA_dd_gg_had_ps_C3_550_PS2.gg.sin"
```

Edit each `.sin` file to 
1) point to the correct circe2 input file
2) have the appropriate sqrts
3) increase the produced number of events to the desired value


Example modified .sin files for C3 can be found in this repo.


Then run whizard for each sin file, e.g. 

```
#Inside WORK_C3_250_PS1
whizard circe_AA_dd_gg_had_ps_C3_250_PS1.gg.sin

```

## Detector simulation for generated events

Once the events have been generated with whizard, we need to pass them through full detector simulation with `ddsim`.

To run ddsim, first download the steering file from [here](https://github.com/Cool-Copper-Collider-Detector-Software/ddsim_workflow/blob/main/ccc_steer_v2.py). Then run the following:

```
#inside proper env, e.g. source setup_EL9.sh for S3DF
source /cvmfs/sw.hsf.org/key4hep/setup.sh
cd $WORK_C3_250_PS1

ddsim --compactFile ${K4GEO}/SiD/compact/SiD_o2_v04/SiD_o2_v04.xml \
      --outputFile ddsim_C3_250_PS1_gg_had.edm4hep.root \
      --steeringFile ccc_steer_v2.py \
      --inputFiles circe_AA_dd_ps_nofilter_00_new.slcio  \
      --vertexOffset 0 0 0 0 \
      --numberOfEvents 100000 \
```



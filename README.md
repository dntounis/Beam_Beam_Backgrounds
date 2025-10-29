# GuineaPig_ddsim

Steering files and batch payloads for generating GuineaPig beam backgrounds and simulating them with `ddsim` for the SiD detector at C^3 energies. The repository is split into two working areas:

- `GuineaPig/` holds Slurm submission helpers and payload scripts that run the `guinea` executable.
- `ddsim/` holds Slurm submission helpers, steering files, and merge utilities for `ddsim` runs.

## Prerequisites

- A personal work area on S3DF (or similar) with Slurm access; update every script to use your own account, email, scratch paths, and output directories.
- Key4HEP nightly environment via CVMFS (`/cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh`) so that `ddsim`, `k4geo`, and `edm4hep` are available.
- A local build of [GuineaPig](https://gitlab.cern.ch/clic-software/guinea-pig) in its own directory (follow the upstream README to configure and compile into `build/bin/guinea` before using these payloads).
- ATLAS environment tools if you need `setupATLAS`; comment out any `setupATLAS` usage if it is not required on your site.

## Environment setup

The scripts assume that the C^3 background workspace lives under your home area and that Key4HEP is available through CVMFS. Adjust paths before running:

```bash
# Example session (adapt to your username, paths, and site requirements)
export HOME=/fs/ddn/sdf/group/atlas/YOUR_ID
setupATLAS -c el9 -b                     # optional; only if your site requires it
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh

cd ~/C^3/bkg_studies_2023/GuineaPig_ddsim_GIT/GuineaPig_ddsim
```

## Generating backgrounds with GuineaPig

1. Build GuineaPig separately and confirm that `GuineaPig/build/bin/guinea` exists.
2. Review `GuineaPig/myJobPayload.sh` and verify the input acceptance file, the collider label (`C3_250_PS1`, `C3_550_PS2`, …), and the output folder layout (`output_new/<collider>/...`). Update paths so they point to your workspace.
3. For a quick local check, run `./myJobPayload.sh C3_250_PS1 400945` from inside `GuineaPig/`. This creates a seed-specific working directory, updates the `rndm_seed` inside the acceptance file, and saves the beam and background products under `output_new/<collider>/`.
4. To launch a production batch on S3DF, edit `GuineaPig/GP_S3DF_SLURM_submission.sh`:
   - Replace `--account`, `--partition`, `--mail-user`, and other Slurm directives with values that match your quota.
   - Uncomment and fill the `seeds=(...)` array with the seeds you want to process; update the `#SBATCH --array` range accordingly.
   - Adjust the `collider` argument in the last `sbatch` command if you duplicate the script for another beam setup.
   Submit with `sbatch GP_S3DF_SLURM_submission.sh`. Slurm will dispatch one array element per seed and call `myJobPayload.sh <collider> <seed>`.

## Simulating with ddsim

1. Inspect `ddsim/myJobPayload_ddsim_*.sh` for the collider configuration you need (250 GeV, 550 GeV, merge jobs, etc.). Update the workspace path (`C^3/bkg_studies_2023/GuineaPig_July_2024` in the checked-in version) and make sure the `output_new` directory structure matches what you produced with GuineaPig.
2. Before launching, convert the background pair files from `.dat` to `.pairs` (the payload scripts perform a simple `cp` with a new extension, but confirm the filenames).
3. Test locally with a single seed, for example:

```bash
ddsim --compactFile ${K4GEO}/SiD/compact/SiD_o2_v04/SiD_o2_v04.xml \
      --outputFile ddsim_C3_250_PS1_seed_400945.edm4hep.root \
      --steeringFile ddsim/ccc_steer_v2.py \
      --inputFiles output_new/C3_250_PS1/bkg_particles/pairs_C3_250_PS1_seed_400945.pairs \
      --vertexOffset 0 0 0 0 \
      --numberOfEvents 1 \
      --guineapig.particlesPerEvent -1
```

Adjust the steering file (`ccc_steer_v2.py` vs `CLICSid.py`) and the output format (`edm4hep.root` vs `.slcio`) as required.

4. For batch production, use the Slurm wrappers in `ddsim/`:
   - `DDSIM_S3DF_SLURM_submission_250.sh` and `myJobPayload_ddsim_250*.sh` cover 250 GeV runs.
   - `DDSIM_S3DF_SLURM_submission_550.sh` and `myJobPayload_ddsim_550*.sh` cover 550 GeV runs.
   - Update each script’s Slurm headers (account, mail, runtime, array range) and confirm the collider argument passed to the payload.
   Submit with `sbatch DDSIM_S3DF_SLURM_submission_XXX.sh`. The payload copies the `.pairs` file, runs `ddsim`, moves the EDM4hep output into `output_new/<collider>/ddsim/`, and removes the temporary `.pairs`.

## Merging ddsim outputs

- `ddsim/myJobPayload_ddsim_550_MERGE_edm4hep.sh` and `ddsim/myJobPayload_ddsim_550_MERGE_slcio.sh` merge partial outputs per seed. They expect the partial files to be present in `output_new/<collider>/ddsim/`.
- `ddsim/DDSIM_S3DF_SLURM_submission_550_MERGE.sh` is an example Slurm launcher that processes the merge tasks in batch mode; update its Slurm directives and seed list the same way as the main production scripts.

## Customisation notes

- All checked-in paths point to the author’s S3DF area—replace them with your own directories before running anything.
- Confirm that the `collider` label you pass on the command line matches both your `output_new/<collider>` hierarchy and the physics configuration (PS1 vs PS2).
- If you change the output format (`.edm4hep.root` vs `.slcio`), edit the payload to update both the `ddsim --outputFile` argument and the final `mv` command.
- Keep the acceptance file (`acc_*.dat`) under version control or back it up; the payload scripts patch the random seed in-place.
- Monitor Slurm logs (`*.out`, `*.err`) and clean up temporary working directories created per seed to save quota.

# dd4hep_hit_analysis_framework
Framework for analyzing dd4hep hit files


Install using:

```
python setup.py install
pip install -e .

```

Simple running scripts to test:

```
python analysis_scripts/simple_plotting.py
python analysis_scripts/simple_occupancy_analysis.py
python analysis_scripts/occupancy_analysis.py

```

To-do list:

- debug occupancy for all subdetectors (currently only working for Vertex Barrel and Endcap)
- timing distribution - study effect of back-scattering
- Study hit distributions in different time buckets -- deconvolve effects of prompt and backscattered hits
- Investigate the use of energy thresholds for occupancy calculation
- For the calorimeters, study the average/total deposited energy of hits per layer and as a function of z

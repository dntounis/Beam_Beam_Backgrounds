#!/bin/bash
# This is the payload script that runs the actual job commands

collider=$1
seed=$2

echo "Running with collider: $collider and seed: $seed"

#sed -i  "s/rndm_seed=[0-9]\+/rndm_seed=$seed/g"  ../testing/acc_C3_final_Summer_2024.dat


# create temp directory to save files to avoid overwriting

#mkdir ${collider}_${seed} && cd ${collider}_${seed} && cp ../build/bin/guinea .
if [ ! -d "${collider}_${seed}" ]; then
    mkdir "${collider}_${seed}"
fi
cd "${collider}_${seed}" && cp ../build/bin/guinea .

cp ../testing/acc_C3_final_Summer_2024.dat acc_C3_final_Summer_2024_${collider}_${seed}.dat

sed -i  "s/rndm_seed=[0-9]\+/rndm_seed=$seed/g"  acc_C3_final_Summer_2024_${collider}_${seed}.dat

./guinea --acc_file acc_C3_final_Summer_2024_${collider}_${seed}.dat "$collider" Jim_pars_Oct2024 ../output_new/"$collider"/test_"$collider"_seed_"$seed".ref

cp lumi.ee.out ../output_new/"$collider"/lumi/lumi_ee_"$collider"_seed_"$seed".out
cp lumi.eg.out ../output_new/"$collider"/lumi/lumi_eg_"$collider"_seed_"$seed".out
cp lumi.ge.out ../output_new/"$collider"/lumi/lumi_ge_"$collider"_seed_"$seed".out
cp lumi.gg.out ../output_new/"$collider"/lumi/lumi_gg_"$collider"_seed_"$seed".out

# changed GP executable to add seed as output to pairs
cp pairs_"$seed".dat ../output_new/"$collider"/bkg_particles/pairs_"$collider"_seed_"$seed".dat
cp tri1.dat ../output_new/"$collider"/bkg_particles/tri1_"$collider"seed_"$seed".dat
cp tri2.dat ../output_new/"$collider"/bkg_particles/tri2_"$collider"seed_"$seed".dat
cp photon.dat ../output_new/"$collider"/bkg_particles/photon_"$collider"seed_"$seed".dat
cp muons.dat ../output_new/"$collider"/bkg_particles/muons_"$collider"seed_"$seed".dat
cp minijet.dat ../output_new/"$collider"/bkg_particles/minijet_"$collider"seed_"$seed".dat
cp hadron.dat ../output_new/"$collider"/bkg_particles/hadron_"$collider"seed_"$seed".dat
cp coh1.dat ../output_new/"$collider"/bkg_particles/coh1_"$collider"seed_"$seed".dat
cp coh2.dat ../output_new/"$collider"/bkg_particles/coh2_"$collider"seed_"$seed".dat

cp beam1.dat ../output_new/"$collider"/beams/beam1_"$collider"seed_"$seed".dat
cp beam2.dat ../output_new/"$collider"/beams/beam2_"$collider"seed_"$seed".dat

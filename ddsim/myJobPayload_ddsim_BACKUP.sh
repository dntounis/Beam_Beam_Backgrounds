#!/bin/bash
# This is the payload script that runs the actual job commands

collider=$1
seed=$2

echo "Running with collider: $collider and seed: $seed"

cd C^3/bkg_studies_2023/GuineaPig_July_2024
#cd /fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024

source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh


input_file="output_new/${collider}/bkg_particles/pairs_${collider}_seed_${seed}.dat"

# add check if energy is 550GeV. If yes, we need to split the input pairs file in two, because the number of particles is too large
# and ddsim runs out of memory

if [[ "$collider" == "C3_550_PS1" || "$collider" == "C3_550_PS2" ]]; then
    # Define output files for split data
    output_file_A="output_new/${collider}/bkg_particles/pairs_${collider}_seed_${seed}_A.pairs"
    output_file_B="output_new/${collider}/bkg_particles/pairs_${collider}_seed_${seed}_B.pairs"
    output_file_C="output_new/${collider}/bkg_particles/pairs_${collider}_seed_${seed}_C.pairs"
    output_file_D="output_new/${collider}/bkg_particles/pairs_${collider}_seed_${seed}_D.pairs"


    # Shuffle the lines randomly and split into four files
    shuf "$input_file" | awk 'NR%4==1{print > "'"$output_file_A"'"} NR%4==2{print > "'"$output_file_B"'"} NR%4==3{print > "'"$output_file_C"'"} NR%4==0{print > "'"$output_file_D"'"}'

else
    # Standard copy if collider is not C3_550_PS1 or PS2
    cp "$input_file" "output_new/${collider}/bkg_particles/pairs_${collider}_seed_${seed}.pairs"
fi

#ddsim --compactFile ${K4GEO}/SiD/compact/SiD_o2_v04/SiD_o2_v04.xml \
#      --outputFile ddsim_${collider}_seed_${seed}.edm4hep.root \
#      --steeringFile ccc_steer.py \
#      --inputFiles output_new/${collider}/bkg_particles/pairs_${collider}_seed_${seed}.pairs \
#      --vertexOffset 0 0 0 0 \
#      --numberOfEvents 1 \
#      --guineapig.particlesPerEvent -1

if [[ "$collider" == "C3_550_PS1" || "$collider" == "C3_550_PS2" ]]; then

        ddsim --compactFile ${K4GEO}/SiD/compact/SiD_o2_v04/SiD_o2_v04.xml \
              --outputFile ddsim_${collider}_seed_${seed}_A.slcio \
              --steeringFile ccc_steer.py \
              --inputFiles output_new/${collider}/bkg_particles/pairs_${collider}_seed_${seed}_A.pairs \
              --vertexOffset 0 0 0 0 \
              --numberOfEvents 1 \
              --guineapig.particlesPerEvent -1

        ddsim --compactFile ${K4GEO}/SiD/compact/SiD_o2_v04/SiD_o2_v04.xml \
              --outputFile ddsim_${collider}_seed_${seed}_B.slcio \
              --steeringFile ccc_steer.py \
              --inputFiles output_new/${collider}/bkg_particles/pairs_${collider}_seed_${seed}_B.pairs \
              --vertexOffset 0 0 0 0 \
              --numberOfEvents 1 \
              --guineapig.particlesPerEvent -1

        ddsim --compactFile ${K4GEO}/SiD/compact/SiD_o2_v04/SiD_o2_v04.xml \
              --outputFile ddsim_${collider}_seed_${seed}_C.slcio \
              --steeringFile ccc_steer.py \
              --inputFiles output_new/${collider}/bkg_particles/pairs_${collider}_seed_${seed}_C.pairs \
              --vertexOffset 0 0 0 0 \
              --numberOfEvents 1 \
              --guineapig.particlesPerEvent -1


	lcio_merge_files ddsim_${collider}_seed_${seed}.slcio ddsim_${collider}_seed_${seed}_A.slcio ddsim_${collider}_seed_${seed}_B.slcio ddsim_${collider}_seed_${seed}_C.slcio
	rm ddsim_${collider}_seed_${seed}_A.slcio && rm ddsim_${collider}_seed_${seed}_B.slcio && rm ddsim_${collider}_seed_${seed}_C.slcio 




else
	ddsim --compactFile ${K4GEO}/SiD/compact/SiD_o2_v04/SiD_o2_v04.xml \
      	      --outputFile ddsim_${collider}_seed_${seed}.slcio \
              --steeringFile ccc_steer.py \
              --inputFiles output_new/${collider}/bkg_particles/pairs_${collider}_seed_${seed}.pairs \
              --vertexOffset 0 0 0 0 \
              --numberOfEvents 1 \
              --guineapig.particlesPerEvent -1
fi




#ddsim --compactFile ${K4GEO}/SiD/compact/SiD_o2_v04/SiD_o2_v04.xml \
#      --outputFile ddsim_${collider}_seed_${seed}.edm4hep.root \
#      --steeringFile ccc_steer.py \
#      --inputFiles pairs_999153.pairs \
#      --vertexOffset 0 0 0 0 \
#      --numberOfEvents 1 \
#      --guineapig.particlesPerEvent -1


#mv ddsim_${collider}_seed_${seed}.edm4hep.root output_new/${collider}/ddsim/ddsim_${collider}_seed_${seed}.edm4hep.root
mv ddsim_${collider}_seed_${seed}.slcio output_new/${collider}/ddsim/ddsim_${collider}_seed_${seed}.slcio


# remove .pairs file to save space (original .dat file is preserved)

import uproot
import numpy as np 
import hist
import math
import mplhep as hep
import xml.etree.ElementTree as ET
from matplotlib.colors import LogNorm
import matplotlib as mpl
import matplotlib.pyplot as plt
import awkward as ak
from collections import Counter,defaultdict
import traceback

from src.hit_analysis.read_hits import read_hits



# List of seeds
#seeds = [527875,733579,360177,286038,479728]  # Replace with your actual list of seeds

# First batch of seeds - 60,
seeds_batch_1 = [
    646726, 824897, 685736, 268021, 893161, 473632, 570691, 973731, 172660, 588225, 
    834692, 765514, 387394, 382466, 893505, 682500, 107626, 581802, 223639, 907367, 
    168136, 116898, 878488, 974354, 429435, 514355, 595020, 298860, 318192, 795000, 
    516831, 938118, 219951, 473269, 620860, 549738, 201099, 969835, 328981, 649096, 
    600364, 495079, 860906, 277515, 883098, 789080, 208830, 751176, 826856, 521374, 
    446610, 442571, 977237, 956193, 638935, 629828, 946409, 711382, 838586, 478805
]

# Second batch of seeds - 73
seeds_batch_2 = [
    568241, 361733, 718355, 670236, 392403, 523638, 775506, 619237, 887397, 796413, 
    307313, 932023, 665869, 180940, 981910, 992834, 541986, 635763, 858530, 361884, 
    501895, 527488, 561072, 982265, 586359, 982832, 743694, 330631, 679469, 558889, 
    749209, 573285, 520380, 994774, 619288, 307004, 368504, 910818, 138965, 443531, 
    928724, 947691, 901624, 393914, 177378, 321364, 599380, 739432, 966089, 815377, 
    733980, 797286, 784389, 807508, 291957, 140752, 800660, 736089, 183876, 409570, 
    779944, 544875, 939767, 428507, 147203, 734028, 511643, 671095, 561651, 774906, 
    724329, 631167, 596798
]

# Third batch of seeds - 60
seeds_batch_3 = [
    527875, 733579, 360177, 286038, 479728, 750622, 869004, 616205, 664707, 132661, 
    781198, 426737, 328457, 144480, 621961, 768697, 685339, 419765, 809804, 564915, 
    262728, 985882, 256863, 353443, 409948, 725103, 996117, 115337, 182286, 909242, 
    984689, 999153, 696557, 949647, 890767, 228995, 677476, 584121, 730711, 321485, 
    446871, 447608, 771760, 190219, 128353, 536212, 886981, 711233, 790240, 577473, 
    524857, 886026, 376225, 440800, 831544, 541993, 576716, 106733, 876089, 754625
]

# Fourth batch of seeds - 73
seeds_batch_4 = [
    340252, 566577, 438336, 983321, 362111, 175226, 711267, 185912, 312192, 847048, 
    400945, 561096, 271634, 831467, 711631, 875591, 203540, 473757, 226527, 595212, 
    917314, 256322, 434222, 840790, 656615, 150344, 255367, 164750, 713978, 431949, 
    804069, 481013, 901407, 342973, 476156, 736050, 469163, 457657, 934359, 664781, 
    428914, 998147, 265876, 614237, 897014, 939699, 367571, 894524, 133257, 323854, 
    984156, 759221, 779288, 124405, 474231, 183930, 232355, 597546, 426705, 311767, 
    919859, 646517, 471934, 880243, 142979, 919932, 659688, 928661, 455310, 975581, 
    506428, 745883, 539398
]


seeds = [seeds_batch_1,seeds_batch_2,seeds_batch_3,seeds_batch_4]


#seeds = [seeds_batch_1,seeds_batch_2]
#seeds = [seeds_batch_1]
seeds = ak.flatten(seeds)

#seeds=[802239]

print(len(seeds))


# Directory path and filename pattern
base_path = "/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/output_new/C3_250_PS1/ddsim/"
filename_pattern = "ddsim_C3_250_PS1_seed_{}.edm4hep.root"

#base_path = "/fs/ddn/sdf/group/atlas/d/dntounis/C^3/bkg_studies_2023/GuineaPig_July_2024/output_new/C3_550_PS1/ddsim/"
#filename_pattern = "ddsim_C3_550_PS1_seed_{}_MERGED.edm4hep.root"


# Open all files with different seeds and store them in a list
events_trees = [uproot.open(base_path + filename_pattern.format(seed)+ ':events') for seed in seeds]


# Now 'files' contains the opened root files for each seed

events_trees[0].keys()


subdetectors_names = ["BeamCalHits", "ECalBarrelHits", "ECalEndcapHits", "HCalBarrelHits", "HCalEndcapHits", "LumiCalHits", "MuonBarrelHits", "MuonEndcapHits", "SiTrackerBarrelHits", "SiTrackerEndcapHits", "SiTrackerForwardHits", "SiVertexBarrelHits", "SiVertexEndcapHits"]
subdetectors_rmax = [140,1600,1350,2650,1550,280,6150,6150,1300,1300,200,70,80]
subdetectors_zmax = [3200,1900,1900,3150,3100,1850,3150,6150,3150,1700,1000,75,200]

# i=0 -> BeamCalHits
# i=1 -> ECalBarrelHits
# i=2 -> ECalEndcapHits
# i=3 -> HCalBarrelHits
# i=4 -> HCalEndcapHits
# i=5 -> LumiCalHits
# i=6 -> MuonBarrelHits
# i=7 -> MuonEndcapHits
# i=8 -> SiTrackerBarrelHits
# i=9 -> SiTrackerEndcapHits
# i=10 -> SiTrackerForwardHits
# i=11 -> SiVertexBarrelHits
# i=12 -> SiVertexEndcapHits

n_phis = [4712, 7226, 11310, 15268, 19163]
n_z = 6300
n_zs = [n_z for i in range(5)]
z_size = 62.5

layers = [
    [0, 13],
    [13, 17],
    [21, 25],
    [34, 38],
    [46.6, 50.6],
    [59, 63],
]


def make_occu_hist(n_phi, n_z, z_size):
    return (
        hist.Hist.new
        .Reg(n_phi, -math.pi, math.pi, name="phi", label="phi")
        .Reg(n_z, -z_size, z_size, name="z", label="z (mm)")
        .Int64()
    )

occu_hists = [
    make_occu_hist(n_phis[i], n_z, z_size) for i in range(5)
]
occu_hists = [make_occu_hist(n_phis[0], n_z, z_size)] + occu_hists

print(occu_hists)

#occu_hist = (
#    hist.Hist.new
#    .Var([0, 13, 19, 29.5, 42.3, 54.8, 78], name="r", label="radius (mm)")
#    .Reg(n_phis[0], -math.pi, math.pi, name="phi", label="phi")
#    .Reg(n_z, -n_z, n_z, name="z", label="z (mm)")
#).Int64()


print(layers)
print(n_zs)

i = 11 #SiVertexBarrelHits
x_combined, y_combined,z_combined,phi_combined,r_combined= read_hits(seeds,events_trees,subdetectors_names[i])
#print(r_combined)

for layer in layers[1:]:
    cut = (r_combined > layer[0]) & (r_combined < layer[1])
    print("r = ", ak.min(layer), " - ", ak.max(layer), "hits in layer = ", sum(cut))
    print(r_combined[cut])


for layer in range(1, 6):
    occu_hists[layer].reset()
    
    
    print("layer =", layer, "rmin =",layers[layer][0], "rmax = ",layers[layer][1])
    cut = (r_combined >= layers[layer][0]) & (r_combined <= layers[layer][1])
    #cut = ak.to_numpy((r_combined >= layers[layer][0]) & (r_combined <= layers[layer][1])
    #print("Cut array size:", ak.num(ak.flatten(phi_combined[cut])), ak.num(ak.flatten(z_combined[cut])))
    print("phi_combined[cut] type:", ak.type(phi_combined[cut]))
    print(sum(cut))
    
    occu_hists[layer].fill(phi_combined[cut],z_combined[cut])
    #occu_hists[layer].fill(ak.flatten(phi_combined[cut]), ak.flatten(z_combined[cut]))
    
   
# 4712*6300=29,685,600    

print("Total bins in histogram:", occu_hists[1].values().size)
print("Total bin contents (hits):", np.sum(occu_hists[1].values()))
print("Total bin contents (hits):", np.sum(occu_hists[1].values()))
print("Number of bins with zero hits:", np.sum(occu_hists[1].values()==0), " or ", np.sum(occu_hists[1].values()==0)/occu_hists[1].values().size*100, "%" )
print("Number of bins with >=1 hits:", np.sum(occu_hists[1].values()>=1), " or ", np.sum(occu_hists[1].values()>=1)/occu_hists[1].values().size*100, "%" )
print("Number of bins with >=2 hits:", np.sum(occu_hists[1].values()>=2), " or ", np.sum(occu_hists[1].values()>=2)/occu_hists[1].values().size*100, "%" )
print("Number of bins with >=3 hits:", np.sum(occu_hists[1].values()>=3), " or ", np.sum(occu_hists[1].values()>=3)/occu_hists[1].values().size*100, "%" )
print("Number of bins with >=4 hits:", np.sum(occu_hists[1].values()>=4), " or ", np.sum(occu_hists[1].values()>=4)/occu_hists[1].values().size*100, "%" )




#print(np.sum(occu_hists[1].values()))
#print(np.sum(occu_hists[1].values()==1))
#print(np.sum(occu_hists[1].values()>1))
#print(np.sum(occu_hists[1].values()==0)+np.sum(occu_hists[1].values()>=1))

def calc_layer_occupancy(ohist, depth, n_train, n_events, n_phi, n_z):
    #return np.sum(ohist.values() >= depth) * n_train / n_events / (n_phi * n_z)
    return np.sum(ohist.values() >= depth)/ohist.values().size/(n_train)

def calc_layer_occupancy_old(ohist, layer, depth, n_train, n_events, n_phi, n_z):
    return np.sum(ohist[layer,:,:].values() >= depth) * n_train / n_events / (n_phi * n_z)

print(len(r_combined))


results = np.zeros((11, 6))

for depth in range(1,11):
    for layer in range(1,6):
        #nz = nz_layer1 if layer == 1 else nz_others
        nz = n_z
        results[depth, layer] = calc_layer_occupancy(
            occu_hists[layer],
            depth,
            1, # one train worth of BXs for sust. scenario
            len(r_combined),
            n_phis[layer-1],
            n_zs[layer-1]
        )
        print(
            depth,
            layer, ":",
            results[depth, layer]
        )


#hep.style.use("CMS")
plt.style.use(hep.style.CMS)

fig, ax = plt.subplots(1)

#print(results[:, 1])

ax.set_ylim((5e-14, 5e-2))
ax.set_yscale('log')
ax.set_ylabel("Fraction of dead cells")
ax.set_xlabel("Assumed buffer depth")
for ilay in range(1, 6):
    ax.errorbar(np.arange(1, 11), results[1:, ilay], xerr=0.5, marker=".", linestyle='', label=f"Layer {ilay}")
ax.set_xticks(np.arange(1, 11))

locmin = mpl.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
ax.yaxis.get_major_locator().set_params(numticks=99)
ax.yaxis.get_minor_locator().set_params(numticks=99, subs=0.1*np.arange(1, 11))

ax.legend()

plt.show()
fig.savefig("simple_occupancy.pdf")

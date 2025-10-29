import matplotlib.pyplot as plt
import hist as hist
import mplhep as hep
from matplotlib.colors import LogNorm
from src.hit_analysis.read_hits import read_hits



def plot_full_detector_with_hits(seeds, subdetectors_names, subdetectors_rmax, subdetectors_zmax, events_trees):
    Ndet = len(subdetectors_names)
    rmax = max(subdetectors_rmax)
    zmax = max(subdetectors_zmax)
    print("rmax = ", rmax, " mm")
    print("zmax = ", zmax, " mm")

    # Create histogram
    combined_histogram = (
        hist.Hist.new.Reg(2000, -zmax, zmax, name="z")
        .Reg(1000, 0, rmax, name="r")
        .Int64()
    )

    # Fill histogram
    for i in range(len(subdetectors_names)):
        print(i, " : ", subdetectors_names[i])
        x_combined, y_combined, z_combined, phi_combined, r_combined = read_hits(seeds, events_trees, subdetectors_names[i])
        combined_histogram.fill(z=z_combined, r=r_combined)

    # Create figure with a single axes
    fig, ax = plt.subplots(figsize=(20, 10))
    #plt.style.use(hep.style.LHCb2)



    # Extract data from histogram
    values = combined_histogram.values()
    z_edges = combined_histogram.axes[0].edges
    r_edges = combined_histogram.axes[1].edges
    
    # Create mesh plot manually 
    mesh = ax.pcolormesh(z_edges, r_edges, values.T, 
                         cmap="plasma", norm=LogNorm())
    
    # Add a single colorbar with improved formatting
    cbar = fig.colorbar(mesh, ax=ax, pad=0.01)  # Adjust padding to control spacing
    cbar.set_label('Hits', fontsize=20, labelpad=15)  # Increase font size and padding
    
    # Set axis labels
    ax.set_xlabel('z [mm]', fontsize=20)
    ax.set_ylabel('r [mm]', fontsize=20)

    # Create a single title line (with Preliminary in bold+italic)
    fig.text(0.22, 0.97, 'Beam-Induced Background Detector Hits', 
             fontsize=24, ha='center')
    
    # Add the second part of the title
    fig.text(0.64, 0.97, r'SiD_o2_v04@C$^{3}$-550 (266 bunches) - ${\it{Preliminary}}$', 
             fontsize=24, ha='center')
    
    # Adjust layout to make room for titles
    plt.tight_layout()
    
    plt.show()
    fig.savefig("SiD_hits_550.png", dpi=300)
    fig.savefig("SiD_hits_550.pdf")
import awkward as ak
import numpy as np


def read_hits(seeds,events_trees,subdetector):
    
    
        arrays = [
            events_tree.arrays([
                f'{subdetector}/{subdetector}.position.x',
                f'{subdetector}/{subdetector}.position.y',
                f'{subdetector}/{subdetector}.position.z'
            ])
            for events_tree in events_trees
        ]


        x = []
        y = []
        z = []


        for i in range(len(seeds)):
            x.append(arrays[i][f'{subdetector}/{subdetector}.position.x'])
            y.append(arrays[i][f'{subdetector}/{subdetector}.position.y'])
            z.append(arrays[i][f'{subdetector}/{subdetector}.position.z'])


        x_combined = ak.concatenate([ak.flatten(array) for array in x])
        y_combined = ak.concatenate([ak.flatten(array) for array in y])   
        z_combined = ak.concatenate([ak.flatten(array) for array in z])


        phi_combined = np.arctan2(x_combined,y_combined)
        r_combined = np.sqrt(x_combined**2 + y_combined**2)

        return x_combined, y_combined,z_combined,phi_combined,r_combined

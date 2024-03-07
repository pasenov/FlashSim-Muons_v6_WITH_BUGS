import numpy as np
import h5py

hf = h5py.File('/home/users/asenov/FlashSim/FlashSim-Muons_v6/efficiencies/dataset/GenMuons.hdf5', 'r')
n1 = np.array(hf["data"][:]) #dataset_name is same as hdf5 object name 

print(n1)

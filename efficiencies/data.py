import os
import sys
import h5py
import torch
import pandas as pd
import uproot
import awkward as ak
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler, Normalizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
file0 = "/home/users/asenov/FlashSim/FlashSim-Muons_v6/extraction/MMuonsA.root" # Change this with your path_name and file_name

from columns import eff_muon

class isReco_Dataset_val(torch.utils.data.Dataset):
    def __init__(self, filepath, input_dim, start, stop):
        h5py_file = h5py.File(filepath, "r")
        self.X = torch.tensor(
            h5py_file["data"][start : (start + stop), 0:(input_dim)],
            dtype=torch.float32,
        )
        self.y = torch.tensor(
            h5py_file["data"][start : (start + stop), -1], dtype=torch.float32
        ).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# class isReco_Dataset(torch.utils.data.Dataset):
#     def __init__(self, filepath, input_dim, start, stop):
#         h5py_file = h5py.File(filepath, "r")
        
#         # Extract X and y from HDF5 file
#         X = h5py_file["data"][start : (start + stop), 0:(input_dim)]
#         y = h5py_file["data"][start : (start + stop), -1]
        
#         # Apply StandardScaler to X
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
        
#         # Convert to torch tensors
#         self.X = torch.tensor(X_scaled, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
    
class isReco_Dataset(torch.utils.data.Dataset):
    def __init__(self, filepath, input_dim, start, stop):
        h5py_file = h5py.File(filepath, "r")
        
        # Extract X and y from HDF5 file
        X = h5py_file["data"][start : (start + stop), 0:(input_dim)]
        y = h5py_file["data"][start : (start + stop), -1]
        
        # Indices of columns to scale
        columns_to_scale_indices = [0, 1, 2, 19, 20, 21, 22, 23]  # Assuming these are the indices of the specified columns
        
        # Apply StandardScaler to specific columns
        scaler = StandardScaler()
        X_scaled = X.copy()
        X_scaled[:, columns_to_scale_indices] = scaler.fit_transform(X_scaled[:, columns_to_scale_indices])
        
        # Convert to torch tensors
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_pd_dataframe(tree, cols, *args, **kwargs):
    df = (
        ak.to_dataframe(tree.arrays(expressions=cols, library="ak", *args, **kwargs))
        .reset_index(drop=True)
        .astype("float32")
        .dropna()
    )
    return df


def dataset_from_root(file_path, cols, name, *args, **kwargs):
    file = uproot.open(file_path, num_workers=20)
    tree = file["MMuons"]
    df = make_pd_dataframe(tree, cols, *args, **kwargs)

    print(df.columns, df.shape)

    f = h5py.File(
        os.path.join(os.path.dirname(__file__), "dataset", f"{name}.hdf5"), "w"
    )
    f.create_dataset("data", data=df.values, dtype="float32")
    f.close()

dataset_from_root(file0, eff_muon, "GenMuons")


import os
<<<<<<< HEAD
=======
import sys

>>>>>>> 98a1361 (commit for efficiency scale)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"

import h5py

import numpy as np

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.optim.lr_scheduler import StepLR


import seaborn as sns
from matplotlib import pyplot as plt

from model import train, MuonClassifier
<<<<<<< HEAD
from data import isReco_Dataset, isReco_Dataset_val
from validation import validation, loss_plot, loss_plot_log
=======
from data import isReco_Dataset_StandardScaler, isReco_Dataset_StandardScaler_transform
from validation import validation, loss_plot, loss_plot_log
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler, Normalizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
file0 = "/home/users/asenov/FlashSim/FlashSim-Muons_v6/extraction/MMuonsA.root" # Change this with your path_name and file_name

from columns import eff_muon

>>>>>>> 98a1361 (commit for efficiency scale)
# from preprocessing import normalize

def training_loop(model, input_dim, datapath, train_size, epochs, lr, batch_size, tag):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize model
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=3)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Adjust parameters as needed
    print(
        f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

<<<<<<< HEAD
    # Load data
    train_dataset = isReco_Dataset(datapath, input_dim, 0, train_size)
    # train_dataset = normalize(train_dataset)
    test_dataset = isReco_Dataset(datapath, input_dim, train_size + 10, 800000)
    # test_dataset = normalize(test_dataset)
    validation_dataset = isReco_Dataset_val(datapath, input_dim, train_size + 20 + 800000, 800000)
    # validation_dataset = normalize(validation_dataset)

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Validation size: {len(validation_dataset)}")
=======
    # Scaler
    scaler = StandardScaler()

    # Load data
    train_dataset = isReco_Dataset_StandardScaler(datapath, input_dim, 0, train_size, scaler)
    # h5py_file = h5py.File(datapath, "r")
    # # Extract X from HDF5 file
    # X = h5py_file["data"][0 : train_size, 0:(input_dim)]    
    # # Indices of columns to scale
    # columns_to_scale_indices = [0, 1, 2, 19, 20, 21, 22, 23]   # Assuming these are the indices of the specified columns which are neither flags nor OHE-ed variables
    # X_scaled = X.copy()
    # scaler.fit(X_scaled[:, columns_to_scale_indices])  # Fit on training data
    # train_dataset = normalize(train_dataset)
    # test_dataset = isReco_Dataset_StandardScaler_transform(datapath, input_dim, train_size + 10, 800000, scaler)
    # test_dataset = normalize(test_dataset)
    # validation_dataset = isReco_Dataset_StandardScaler_transform(datapath, input_dim, train_size + 20 + 800000, 800000, scaler)
    # validation_dataset = normalize(validation_dataset)

    np.save(os.path.join(train_dataset.save_dir, f"X_scaled.npy"), train_dataset.X.numpy())
    np.save(os.path.join(train_dataset.save_dir, f"scaler_mean.npy"), train_dataset.scaler.mean_)
    np.save(os.path.join(train_dataset.save_dir, f"scaler_std.npy"), train_dataset.scaler.scale_)


    print(f"Train size: {len(train_dataset)}")
    # print(f"Test size: {len(test_dataset)}")
    # print(f"Validation size: {len(validation_dataset)}")
>>>>>>> 98a1361 (commit for efficiency scale)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
<<<<<<< HEAD
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=100000, shuffle=True
    )
=======
    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=True
    # )
    # validation_dataloader = torch.utils.data.DataLoader(
    #     validation_dataset, batch_size=100000, shuffle=True
    # )
>>>>>>> 98a1361 (commit for efficiency scale)

    # Train the model
    test_history = []
    train_history = []

    for epoch in range(epochs):
        print(f"Epoch {(epoch + 1):03}:")
        tr_loss, te_loss = train(
            train_dataloader,
<<<<<<< HEAD
            test_dataloader,
=======
            #test_dataloader,
            None,
>>>>>>> 98a1361 (commit for efficiency scale)
            model,
            loss_fn,
            optimizer,
            scheduler,
            device,
        )
        test_history.append(te_loss)
        train_history.append(tr_loss)

<<<<<<< HEAD
=======
        # if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            # Save the scaled data
            # np.save(os.path.join(train_dataset.save_dir, f"X_scaled_epoch_{epoch}.npy"), train_dataset.X.numpy())
            # np.save(os.path.join(train_dataset.save_dir, f"scaler_mean_epoch_{epoch}.npy"), train_dataset.scaler.mean_)
            # np.save(os.path.join(train_dataset.save_dir, f"scaler_std_epoch_{epoch}.npy"), train_dataset.scaler.scale_)


>>>>>>> 98a1361 (commit for efficiency scale)
        # Plot loss
        loss_plot(train_history, test_history, tag)
        loss_plot_log(train_history, test_history, tag)


        if epoch % 5 == 0:
            # Save the model
            torch.save(
                model.state_dict(),
                os.path.join(
                    os.path.dirname(__file__), "models", f"efficiency_{tag}.pt"
                ),
            )
<<<<<<< HEAD
            # Validation
            validation(
                validation_dataloader=validation_dataloader,
                model=model,
                device=device,
                tag=tag
            )
=======
            # # Validation
            # validation(
            #     validation_dataloader=validation_dataloader,
            #     model=model,
            #     device=device,
            #     tag=tag
            # )
>>>>>>> 98a1361 (commit for efficiency scale)

def GenMuon_efficiency():
    input_dim = 32
    model = MuonClassifier(input_dim)
    datapath = os.path.join(os.path.dirname(__file__), "dataset", "GenMuons.hdf5")
    train_size = 3800000
    # epochs = 2000
    epochs = 250
    lr = 1e-4
    batch_size = 100000
    tag = "muons"

    training_loop(
        model, input_dim, datapath, train_size, epochs, lr, batch_size, tag
    )

print("Training GenMuon classifier...")
GenMuon_efficiency()
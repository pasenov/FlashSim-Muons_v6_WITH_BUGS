import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import torch
from model import MuonClassifier
<<<<<<< HEAD
=======
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler, Normalizer
>>>>>>> 98a1361 (commit for efficiency scale)

np.random.seed(42)

muon_cond = [
    "GenMuon_eta",
    "GenMuon_phi",
    "GenMuon_pt",
    "GenMuon_charge",
    "GenMuon_statusFlag0",
    "GenMuon_statusFlag1",
    "GenMuon_statusFlag2",
    "GenMuon_statusFlag3",
    "GenMuon_statusFlag4",
    "GenMuon_statusFlag5",
    "GenMuon_statusFlag6",
    "GenMuon_statusFlag7",
    "GenMuon_statusFlag8",
    "GenMuon_statusFlag9",
    "GenMuon_statusFlag10",
    "GenMuon_statusFlag11",
    "GenMuon_statusFlag12",
    "GenMuon_statusFlag13",
    "GenMuon_statusFlag14",
    "ClosestJet_dr",
    "ClosestJet_dphi",
    "ClosestJet_deta",
    "ClosestJet_pt",
    "ClosestJet_mass",
    "ClosestJet_EncodedPartonFlavour_light",
    "ClosestJet_EncodedPartonFlavour_gluon",
    "ClosestJet_EncodedPartonFlavour_c",
    "ClosestJet_EncodedPartonFlavour_b",
    "ClosestJet_EncodedPartonFlavour_undefined",
    "ClosestJet_EncodedHadronFlavour_b",
    "ClosestJet_EncodedHadronFlavour_c",
    "ClosestJet_EncodedHadronFlavour_light",
]

eff_muon = muon_cond + ["GenMuon_isReco"]

<<<<<<< HEAD
# open the file
f = h5py.File("dataset/GenMuons.hdf5", "r")

# make a dataframe
df = pd.DataFrame(f["data"][:], columns=eff_muon)
=======
# Define the filepath
filepath = "dataset/GenMuons.hdf5"

# Indices of columns to scale
columns_to_scale_indices = [0, 1, 2, 19, 20, 21, 22, 23]  # Assuming these are the indices of the specified columns which are neither flags nor OHE-ed variables

# Open the HDF5 file
with h5py.File(filepath, "r") as fpp:
    # Load the data
    datapp = np.array(fpp["data"][:])
    print(datapp.shape)
    datapp = datapp[datapp[:, 2]> 20, :]
    print(datapp.shape)

    # Apply StandardScaler to specific columns
    scaler = StandardScaler()
    scaler.mean_ = np.load('scaled_data/scaler_mean.npy')
    scaler.scale_ = np.load('scaled_data/scaler_std.npy')

    datapp[:, columns_to_scale_indices] = scaler.transform(datapp[:, columns_to_scale_indices])

print(datapp.shape)
print(datapp[0])

# make a dataframe
dfpp = pd.DataFrame(datapp, columns=eff_muon) # after preprocessing
print("dfpp = pd.DataFrame(fpp[data][:], columns=eff_muon); len(dfpp) = ", len(dfpp))
# dfpp = dfpp[dfpp["GenMuon_pt"] > 20]
# print("dfpp = dfpp[dfpp[GenMuon_pt] > 20]; len(dfpp) = ", len(dfpp))

# Open the HDF5 file
f = h5py.File("dataset/GenMuons.hdf5", "r")

# make a dataframe
df = pd.DataFrame(f["data"][:], columns=eff_muon) # before preprocessing
>>>>>>> 98a1361 (commit for efficiency scale)
print("df = pd.DataFrame(f[data][:], columns=eff_muon); len(df) = ", len(df))
df = df[df["GenMuon_pt"] > 20]
print("df = df[df[GenMuon_pt] > 20]; len(df) = ", len(df))

<<<<<<< HEAD

# load the model
model = MuonClassifier(32)
model.load_state_dict(torch.load("models/efficiency_muons.pt"))
=======
# load the model
model = MuonClassifier(32)
# model.load_state_dict(torch.load("models/efficiency_muons.pt"))
# model.load_state_dict(torch.load("models/efficiency_muons_ReduceLROnPlateau_patience_3_SGD.pt"))
model.load_state_dict(torch.load("models/efficiency_muons.pt"))

>>>>>>> 98a1361 (commit for efficiency scale)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = model.to(device)
model.eval()

# make the prediction
<<<<<<< HEAD
X = torch.tensor(df[eff_muon[0:-1]].values, dtype=torch.float32).to(device)
y_pred = model.predict(X)
y_pred = y_pred.detach().cpu().numpy().flatten()
p = np.random.rand(y_pred.size)
tmp = np.ones(y_pred.size)
df["isReco"] = np.where(y_pred > p, tmp, 0)

xbins_ = np.linspace(20, 150, 20)
ybins_ = np.linspace(0, 2, 20)

bin_content, xbins, ybins = np.histogram2d(
    df["GenMuon_pt"],
    df["ClosestJet_dr"],
    bins=(xbins_, ybins_),
    range=((20, 150), (0, 2)),
)

bin_content_reco, xbins, ybins = np.histogram2d(
    df["GenMuon_pt"],
    df["ClosestJet_dr"],
    bins=(xbins_, ybins_),
    range=((20, 150), (0, 2)),
    weights=df["isReco"],
)

eff = bin_content_reco / bin_content

full_bin_content_reco, xbins, ybins = np.histogram2d(
    df["GenMuon_pt"],
    df["ClosestJet_dr"],
    bins=(xbins_, ybins_),
    range=((20, 150), (0, 2)),
    weights=df["GenMuon_isReco"],
)

full_eff = full_bin_content_reco / bin_content

# make the plot of the two efficiencies
hep.style.use(hep.style.CMS)
fig, ax = plt.subplots(1, 2, figsize=(30, 15), sharey=True, width_ratios=[1, 1.2])
hep.cms.text("Private Work", loc=0, ax=ax[0])
im = ax[0].imshow(
    eff.T,
    interpolation="none",
    origin="lower",
    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
    aspect="auto",
    cmap="cividis",
    # vmin=0.5,
    vmin=0.0,
    vmax=1,
)
ax[0].set_xlabel(r"$p_{T}^{GEN}$ [GeV]")
ax[0].set_ylabel(r"$\Delta R^{GEN}_{e-jet}$")
ax[0].set_title(r"FlashSim ($p_{T}^{GEN}>20$ GeV)", loc="right")


ax[1].imshow(
    full_eff.T,
    interpolation="none",
    origin="lower",
    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
    aspect="auto",
    cmap="cividis",
    # vmin=0.5,
    vmin=0.0,
    vmax=1,
)
ax[1].set_xlabel(r"$p_{T}^{GEN}$ [GeV]")
ax[1].set_title(r"FullSim ($p_{T}^{GEN}>20$ GeV)", loc="right")


cbar = fig.colorbar(im, ax=ax[1])

plt.savefig("efficiency_pt_dr.pdf")

# the same for GenMuon_eta and GenMuon_phi

xxbins_ = np.linspace(-2.5, 2.5, 20)
yybins_ = np.linspace(-3.14, 3.14, 20)

bin_content, xbins, ybins = np.histogram2d(
    df["GenMuon_eta"],
    df["GenMuon_phi"],
    bins=(xxbins_, yybins_),
    range=((-2.5, 2.5), (-3.14, 3.14)),
)

bin_content_reco, xbins, ybins = np.histogram2d(
    df["GenMuon_eta"],
    df["GenMuon_phi"],
    bins=(xxbins_, yybins_),
    range=((-2.5, 2.5), (-3.14, 3.14)),
    weights=df["isReco"],
)

eff = bin_content_reco / bin_content

full_bin_content_reco, xbins, ybins = np.histogram2d(
    df["GenMuon_eta"],
    df["GenMuon_phi"],
    bins=(xxbins_, yybins_),
    range=((-2.5, 2.5), (-3.14, 3.14)),
    weights=df["GenMuon_isReco"],
)

full_eff = full_bin_content_reco / bin_content

hep.style.use(hep.style.CMS)
fig, ax = plt.subplots(1, 2, figsize=(30, 15), sharey=True, width_ratios=[1, 1.2])
hep.cms.text("Private Work", loc=0, ax=ax[0])
im = ax[0].imshow(
    eff.T,
    interpolation="none",
    origin="lower",
    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
    aspect="auto",
    cmap="cividis",
    vmin=0.8,
    # vmin=0.0,
    vmax=1,
)
ax[0].set_xlabel(r"$\eta^{GEN}$")
ax[0].set_ylabel(r"$\phi^{GEN}$")
ax[0].set_title(r"FlashSim ($p_{T}^{GEN}>20$ GeV)", loc="right")

ax[1].imshow(
    full_eff.T,
    interpolation="none",
    origin="lower",
    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
    aspect="auto",
    cmap="cividis",
    vmin=0.8,
    # vmin=0.0,
    vmax=1,
)
ax[1].set_xlabel(r"$\eta^{GEN}$")
ax[1].set_title(r"FullSim ($p_{T}^{GEN}>20$ GeV)", loc="right")

cbar = fig.colorbar(im, ax=ax[1])

plt.savefig("efficiency_eta_phi.pdf")

# 1d GenMuon_pt

xbins_ = np.linspace(20, 150, 20)

bin_width = 130 / 20 / 2

bin_content, xbins = np.histogram(df["GenMuon_pt"], bins=xbins_, range=(20, 150))

err = np.sqrt(bin_content)

bin_content_reco, xbins = np.histogram(
    df["GenMuon_pt"], bins=xbins_, range=(20, 150), weights=df["isReco"]
)

err_reco = np.sqrt(bin_content_reco)

eff = bin_content_reco / bin_content

yerr = eff * np.sqrt((err_reco / bin_content_reco) ** 2 + (err / bin_content) ** 2)


full_bin_content_reco, xbins = np.histogram(
    df["GenMuon_pt"], bins=xbins_, range=(20, 150), weights=df["GenMuon_isReco"]
)

err_reco_full = np.sqrt(full_bin_content_reco)

full_eff = full_bin_content_reco / bin_content

yerr_full = full_eff * np.sqrt(
    (err_reco_full / full_bin_content_reco) ** 2 + (err / bin_content) ** 2
)

bin_centers = (xbins[1:] + xbins[:-1]) / 2

hep.style.use(hep.style.CMS)
fig, ax = plt.subplots(figsize=(12, 12))
hep.cms.text("Private Work", loc=0, ax=ax)
ax.errorbar(
    bin_centers,
    full_eff,
    yerr=yerr_full,
    xerr=bin_width,
    label="FullSim",
    color="black",
    lw=2,
    ls="",
    fmt="s",
    markersize=10,
    zorder=1,
)
ax.errorbar(
    bin_centers,
    eff,
    yerr=yerr,
    xerr=bin_width,
    label="FlashSim",
    color="orange",
    lw=2,
    ls="",
    fmt="o",
    markersize=6,
    zorder=2,
)

ax.set_xlabel(r"$p_{T}^{GEN}$ [GeV]")
ax.set_ylabel(r"Efficiency")
ax.set_title(r"($p_{T}^{GEN}>20$ GeV)", loc="right")
# ax.set_ylim(0.5, 1)
ax.set_ylim(0.0, 1.1)


ax.legend()

plt.savefig("efficiency_pt.pdf")

# same for GenMuon_eta and GenMuon_phi

xxbins_ = np.linspace(-2.5, 2.5, 20)

bin_width = 2.5 / 20

bin_content, xbins = np.histogram(
    df["GenMuon_eta"], bins=xxbins_, range=(-2.5, 2.5)
)

err = np.sqrt(bin_content)

bin_content_reco, xbins = np.histogram(
    df["GenMuon_eta"], bins=xxbins_, range=(-2.5, 2.5), weights=df["isReco"]
)

err_reco = np.sqrt(bin_content_reco)

eff = bin_content_reco / bin_content

yerr = eff * np.sqrt((err_reco / bin_content_reco) ** 2 + (err / bin_content) ** 2)


full_bin_content_reco, xbins = np.histogram(
    df["GenMuon_eta"],
    bins=xxbins_,
    range=(-2.5, 2.5),
    weights=df["GenMuon_isReco"],
)

err_reco_full = np.sqrt(full_bin_content_reco)

full_eff = full_bin_content_reco / bin_content

yerr_full = full_eff * np.sqrt(
    (err_reco_full / full_bin_content_reco) ** 2 + (err / bin_content) ** 2
)

bin_centers = (xbins[1:] + xbins[:-1]) / 2

hep.style.use(hep.style.CMS)
fig, ax = plt.subplots(figsize=(12, 12))
hep.cms.text("Private Work", loc=0, ax=ax)
ax.errorbar(
    bin_centers,
    full_eff,
    yerr=yerr_full,
    xerr=bin_width,
    label="FullSim",
    color="black",
    lw=2,
    ls="",
    fmt="s",
    markersize=10,
    zorder=1,
)
ax.errorbar(
    bin_centers,
    eff,
    xerr=bin_width,
    yerr=yerr,
    label="FlashSim",
    color="orange",
    lw=2,
    ls="",
    fmt="o",
    markersize=6,
    zorder=2,
)

ax.set_xlabel(r"$\eta^{GEN}$")
ax.set_ylabel(r"Efficiency")
ax.set_title(r"($p_{T}^{GEN}>20$ GeV)", loc="right")
ax.set_ylim(0.0, 1.1)


ax.legend()

plt.savefig("efficiency_eta.pdf")

# same for GenMuon_phi

=======
# tmp = np.load("scaled_data/X_scaled.npy")
# tmp = tmp[tmp[:, 2] > 20] # pT > 20 GeV
# print("tmp(0) = ", tmp[0])
# Xpp = torch.tensor(tmp[:, 0:-1], dtype=torch.float32).to(device)
Xpp = torch.tensor(dfpp.values[:, 0:-1], dtype=torch.float32).to(device)
y_predpp = model.predict(Xpp)
print(y_predpp)
y_predpp = y_predpp.detach().cpu().numpy().flatten()
ppp = np.random.rand(y_predpp.size)
tmppp = np.ones(y_predpp.size)
dfpp["isReco"] = np.where(y_predpp > ppp, tmppp, 0)

print(dfpp["isReco"][dfpp["isReco"] == 1])
print(df["GenMuon_isReco"][df["GenMuon_isReco"] == 1])

# X = torch.tensor(df[eff_muon[0:-1]].values, dtype=torch.float32).to(device)
# y_pred = model.predict(X)
# y_pred = y_pred.detach().cpu().numpy().flatten()
# p = np.random.rand(y_pred.size)
# tmp = np.ones(y_pred.size)
# df["isReco"] = np.where(y_pred > p, tmp, 0)

# xbins_ = np.linspace(20, 150, 20)
# ybins_ = np.linspace(0, 2, 20)

# bin_content, xbins, ybins = np.histogram2d(
#     df["GenMuon_pt"],
#     df["ClosestJet_dr"],
#     bins=(xbins_, ybins_),
#     range=((20, 150), (0, 2)),
# )

# bin_content_reco, xbins, ybins = np.histogram2d(
#     df["GenMuon_pt"],
#     df["ClosestJet_dr"],
#     bins=(xbins_, ybins_),
#     range=((20, 150), (0, 2)),
#     weights=df["isReco"],
# )

# eff = bin_content_reco / bin_content

# full_bin_content_reco, xbins, ybins = np.histogram2d(
#     df["GenMuon_pt"],
#     df["ClosestJet_dr"],
#     bins=(xbins_, ybins_),
#     range=((20, 150), (0, 2)),
#     weights=df["GenMuon_isReco"],
# )

# full_eff = full_bin_content_reco / bin_content

# # make the plot of the two efficiencies
# hep.style.use(hep.style.CMS)
# fig, ax = plt.subplots(1, 2, figsize=(30, 15), sharey=True, width_ratios=[1, 1.2])
# hep.cms.text("Private Work", loc=0, ax=ax[0])
# im = ax[0].imshow(
#     eff.T,
#     interpolation="none",
#     origin="lower",
#     extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
#     aspect="auto",
#     cmap="cividis",
#     # vmin=0.5,
#     vmin=0.0,
#     vmax=1,
# )
# ax[0].set_xlabel(r"$p_{T}^{GEN}$ [GeV]")
# ax[0].set_ylabel(r"$\Delta R^{GEN}_{e-jet}$")
# ax[0].set_title(r"FlashSim ($p_{T}^{GEN}>20$ GeV)", loc="right")


# ax[1].imshow(
#     full_eff.T,
#     interpolation="none",
#     origin="lower",
#     extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
#     aspect="auto",
#     cmap="cividis",
#     # vmin=0.5,
#     vmin=0.0,
#     vmax=1,
# )
# ax[1].set_xlabel(r"$p_{T}^{GEN}$ [GeV]")
# ax[1].set_title(r"FullSim ($p_{T}^{GEN}>20$ GeV)", loc="right")


# cbar = fig.colorbar(im, ax=ax[1])

# plt.savefig("efficiency_pt_dr.pdf")

# the same for GenMuon_eta and GenMuon_phi

# xxbins_ = np.linspace(-2.5, 2.5, 20)
# yybins_ = np.linspace(-3.14, 3.14, 20)

# bin_content, xbins, ybins = np.histogram2d(
#     dfpp["GenMuon_eta"],
#     dfpp["GenMuon_phi"],
#     bins=(xxbins_, yybins_),
#     range=((-2.5, 2.5), (-3.14, 3.14)),
# )

# bin_content_reco, xbins, ybins = np.histogram2d(
#     dfpp["GenMuon_eta"],
#     dfpp["GenMuon_phi"],
#     bins=(xxbins_, yybins_),
#     range=((-2.5, 2.5), (-3.14, 3.14)),
#     weights=dfpp["isReco"],
# )

# eff = bin_content_reco / bin_content

# full_bin_content_reco, xbins, ybins = np.histogram2d(
#     dfpp["GenMuon_eta"],
#     dfpp["GenMuon_phi"],
#     bins=(xxbins_, yybins_),
#     range=((-2.5, 2.5), (-3.14, 3.14)),
#     weights=dfpp["GenMuon_isReco"],
# )

# full_eff = full_bin_content_reco / bin_content

# hep.style.use(hep.style.CMS)
# fig, ax = plt.subplots(1, 2, figsize=(30, 15), sharey=True, width_ratios=[1, 1.2])
# hep.cms.text("Private Work", loc=0, ax=ax[0])
# im = ax[0].imshow(
#     eff.T,
#     interpolation="none",
#     origin="lower",
#     extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
#     aspect="auto",
#     cmap="cividis",
#     # vmin=0.8,
#     vmin=0.0,
#     vmax=1,
# )
# ax[0].set_xlabel(r"$\eta^{GEN}$")
# ax[0].set_ylabel(r"$\phi^{GEN}$")
# ax[0].set_title(r"FlashSim ($p_{T}^{GEN}>20$ GeV)", loc="right")

# ax[1].imshow(
#     full_eff.T,
#     interpolation="none",
#     origin="lower",
#     extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
#     aspect="auto",
#     cmap="cividis",
#     vmin=0.8,
#     # vmin=0.0,
#     vmax=1,
# )
# ax[1].set_xlabel(r"$\eta^{GEN}$")
# ax[1].set_title(r"FullSim ($p_{T}^{GEN}>20$ GeV)", loc="right")

# cbar = fig.colorbar(im, ax=ax[1])

# plt.savefig("efficiency_eta_phi.pdf")

# # 1d GenMuon_pt

# xbins_ = np.linspace(20, 150, 20)

# bin_width = 130 / 20 / 2

# full_bin_content, xbins = np.histogram(dfpp["GenMuon_pt"], bins=xbins_, range=(20, 150))

# err_full = np.sqrt(full_bin_content)

# bin_content, xbins = np.histogram(dfpp["GenMuon_pt"], bins=xbins_, range=(20, 150))

# err = np.sqrt(bin_content)

# bin_content_reco, xbins = np.histogram(
#     dfpp["GenMuon_pt"], bins=xbins_, range=(20, 150), weights=dfpp["isReco"]
# )

# err_reco = np.sqrt(bin_content_reco)

# eff = bin_content_reco / bin_content

# yerr = eff * np.sqrt((err_reco / bin_content_reco) ** 2 + (err / bin_content) ** 2)


# full_bin_content_reco, xbins = np.histogram(
#     dfpp["GenMuon_pt"], bins=xbins_, range=(20, 150), weights=dfpp["GenMuon_isReco"]
# )

# err_reco_full = np.sqrt(full_bin_content_reco)

# full_eff = full_bin_content_reco / bin_content

# yerr_full = full_eff * np.sqrt(
#     (err_reco_full / full_bin_content_reco) ** 2 + (err_full / full_bin_content) ** 2
# )

# bin_centers = (xbins[1:] + xbins[:-1]) / 2

# hep.style.use(hep.style.CMS)
# fig, ax = plt.subplots(figsize=(12, 12))
# hep.cms.text("Private Work", loc=0, ax=ax)
# ax.errorbar(
#     bin_centers,
#     full_eff,
#     yerr=yerr_full,
#     xerr=bin_width,
#     label="FullSim",
#     color="black",
#     lw=2,
#     ls="",
#     fmt="s",
#     markersize=10,
#     zorder=1,
# )
# ax.errorbar(
#     bin_centers,
#     eff,
#     yerr=yerr,
#     xerr=bin_width,
#     label="FlashSim",
#     color="orange",
#     lw=2,
#     ls="",
#     fmt="o",
#     markersize=6,
#     zorder=2,
# )

# ax.set_xlabel(r"$p_{T}^{GEN}$ [GeV]")
# ax.set_ylabel(r"Efficiency")
# ax.set_title(r"($p_{T}^{GEN}>20$ GeV)", loc="right")
# # ax.set_ylim(0.5, 1)
# ax.set_ylim(0.0, 1.1)


# ax.legend()

# plt.savefig("efficiency_pt.pdf")

# # same for GenMuon_eta and GenMuon_phi

# xxbins_ = np.linspace(-2.5, 2.5, 20)

# bin_width = 2.5 / 20

# full_bin_content, xbins = np.histogram(
#     dfpp["GenMuon_eta"], bins=xxbins_, range=(-2.5, 2.5)
# )

# err_full = np.sqrt(full_bin_content)

# bin_content, xbins = np.histogram(
#     dfpp["GenMuon_eta"], bins=xxbins_, range=(-2.5, 2.5)
# )

# err = np.sqrt(bin_content)

# bin_content_reco, xbins = np.histogram(
#     dfpp["GenMuon_eta"], bins=xxbins_, range=(-2.5, 2.5), weights=dfpp["isReco"]
# )

# err_reco = np.sqrt(bin_content_reco)

# eff = bin_content_reco / bin_content

# yerr = eff * np.sqrt((err_reco / bin_content_reco) ** 2 + (err / bin_content) ** 2)


# full_bin_content_reco, xbins = np.histogram(
#     dfpp["GenMuon_eta"],
#     bins=xxbins_,
#     range=(-2.5, 2.5),
#     weights=dfpp["GenMuon_isReco"],
# )

# err_reco_full = np.sqrt(full_bin_content_reco)

# full_eff = full_bin_content_reco / full_bin_content

# yerr_full = full_eff * np.sqrt(
#     (err_reco_full / full_bin_content_reco) ** 2 + (err_full / full_bin_content) ** 2
# )

# bin_centers = (xbins[1:] + xbins[:-1]) / 2

# hep.style.use(hep.style.CMS)
# fig, ax = plt.subplots(figsize=(12, 12))
# hep.cms.text("Private Work", loc=0, ax=ax)
# ax.errorbar(
#     bin_centers,
#     full_eff,
#     yerr=yerr_full,
#     xerr=bin_width,
#     label="FullSim",
#     color="black",
#     lw=2,
#     ls="",
#     fmt="s",
#     markersize=10,
#     zorder=1,
# )
# ax.errorbar(
#     bin_centers,
#     eff,
#     xerr=bin_width,
#     yerr=yerr,
#     label="FlashSim",
#     color="orange",
#     lw=2,
#     ls="",
#     fmt="o",
#     markersize=6,
#     zorder=2,
# )

# ax.set_xlabel(r"$\eta^{GEN}$")
# ax.set_ylabel(r"Efficiency")
# ax.set_title(r"($p_{T}^{GEN}>20$ GeV)", loc="right")
# ax.set_ylim(0.0, 1.1)


# ax.legend()

# plt.savefig("efficiency_eta.pdf")

# same for GenMuon_phi

# xxbins_ = np.linspace(-3.14, 3.14, 20)

# bin_width = 3.14 / 20

# full_bin_content, xbins = np.histogram(
#     dfpp["GenMuon_phi"], bins=xxbins_, range=(-3.14, 3.14)
# )

# err_full = np.sqrt(full_bin_content)

# bin_content, xbins = np.histogram(
#     dfpp["GenMuon_phi"], bins=xxbins_, range=(-3.14, 3.14)
# )

# err = np.sqrt(bin_content)

# bin_content_reco, xbins = np.histogram(
#     dfpp["GenMuon_phi"], bins=xxbins_, range=(-3.14, 3.14), weights=dfpp["isReco"]
# )

# err_reco = np.sqrt(bin_content_reco)

# eff = bin_content_reco / bin_content

# yerr = eff * np.sqrt((err_reco / bin_content_reco) ** 2 + (err / bin_content) ** 2)


# full_bin_content_reco, xbins = np.histogram(
#     dfpp["GenMuon_phi"],
#     bins=xxbins_,
#     range=(-3.14, 3.14),
#     weights=dfpp["GenMuon_isReco"], # df
# )

# err_reco_full = np.sqrt(full_bin_content_reco)

# full_eff = full_bin_content_reco / full_bin_content

# yerr_full = full_eff * np.sqrt(
#     (err_reco_full / full_bin_content_reco) ** 2 + (err_full / full_bin_content) ** 2
# )

# bin_centers = (xbins[1:] + xbins[:-1]) / 2

# hep.style.use(hep.style.CMS)
# fig, ax = plt.subplots(figsize=(12, 12))
# hep.cms.text("Private Work", loc=0, ax=ax)
# ax.errorbar(
#     bin_centers,
#     full_eff,
#     yerr=yerr_full,
#     xerr=bin_width,
#     label="FullSim",
#     color="black",
#     lw=2,
#     ls="",
#     fmt="s",
#     markersize=10,
#     zorder=1,
# )
# ax.errorbar(
#     bin_centers,
#     eff,
#     yerr=yerr,
#     xerr=bin_width,
#     label="FlashSim",
#     color="orange",
#     lw=2,
#     ls="",
#     fmt="o",
#     markersize=6,
#     zorder=2,
# )

# ax.set_xlabel(r"$\phi^{GEN}$")
# ax.set_ylabel(r"Efficiency")
# ax.set_title(r"($p_{T}^{GEN}>20$ GeV)", loc="right")
# # ax.set_ylim(0.5, 1.1)
# ax.set_ylim(0.0, 1.1)

# ax.legend()

# plt.savefig("efficiency_phi.pdf")





# # Inverse transform the scaled data for plotting
# inverse_transformed_datapp = scaler.inverse_transform(dfpp.iloc[:, columns_to_scale_indices])
# df = pd.DataFrame(inverse_transformed_datapp, columns=dfpp.columns[columns_to_scale_indices])


>>>>>>> 98a1361 (commit for efficiency scale)
xxbins_ = np.linspace(-3.14, 3.14, 20)

bin_width = 3.14 / 20

<<<<<<< HEAD
=======
# denominatore

>>>>>>> 98a1361 (commit for efficiency scale)
bin_content, xbins = np.histogram(
    df["GenMuon_phi"], bins=xxbins_, range=(-3.14, 3.14)
)

err = np.sqrt(bin_content)

<<<<<<< HEAD
bin_content_reco, xbins = np.histogram(
    df["GenMuon_phi"], bins=xxbins_, range=(-3.14, 3.14), weights=df["isReco"]
=======
# numerator for FlashSim

bin_content_reco, xbins = np.histogram(
    df["GenMuon_phi"], bins=xxbins_, range=(-3.14, 3.14), weights=dfpp["isReco"]
>>>>>>> 98a1361 (commit for efficiency scale)
)

err_reco = np.sqrt(bin_content_reco)

eff = bin_content_reco / bin_content

yerr = eff * np.sqrt((err_reco / bin_content_reco) ** 2 + (err / bin_content) ** 2)

<<<<<<< HEAD
=======
# numerator for FullSim
>>>>>>> 98a1361 (commit for efficiency scale)

full_bin_content_reco, xbins = np.histogram(
    df["GenMuon_phi"],
    bins=xxbins_,
    range=(-3.14, 3.14),
<<<<<<< HEAD
    weights=df["GenMuon_isReco"],
=======
    weights=df["GenMuon_isReco"], # df
>>>>>>> 98a1361 (commit for efficiency scale)
)

err_reco_full = np.sqrt(full_bin_content_reco)

full_eff = full_bin_content_reco / bin_content

yerr_full = full_eff * np.sqrt(
<<<<<<< HEAD
    (err_reco_full / full_bin_content_reco) ** 2 + (err / bin_content) ** 2
=======
    (err_reco_full / bin_content_reco) ** 2 + (err / bin_content) ** 2
>>>>>>> 98a1361 (commit for efficiency scale)
)

bin_centers = (xbins[1:] + xbins[:-1]) / 2

hep.style.use(hep.style.CMS)
fig, ax = plt.subplots(figsize=(12, 12))
hep.cms.text("Private Work", loc=0, ax=ax)
ax.errorbar(
    bin_centers,
    full_eff,
    yerr=yerr_full,
    xerr=bin_width,
    label="FullSim",
    color="black",
    lw=2,
    ls="",
    fmt="s",
    markersize=10,
    zorder=1,
)
ax.errorbar(
    bin_centers,
    eff,
    yerr=yerr,
    xerr=bin_width,
    label="FlashSim",
    color="orange",
    lw=2,
    ls="",
    fmt="o",
    markersize=6,
    zorder=2,
)

ax.set_xlabel(r"$\phi^{GEN}$")
ax.set_ylabel(r"Efficiency")
ax.set_title(r"($p_{T}^{GEN}>20$ GeV)", loc="right")
<<<<<<< HEAD
ax.set_ylim(0.5, 1.1)

ax.legend()

plt.savefig("efficiency_phi.pdf")
=======
# ax.set_ylim(0.5, 1.1)
ax.set_ylim(0.0, 1.1)

ax.legend()

plt.savefig("efficiency_phi.pdf")
>>>>>>> 98a1361 (commit for efficiency scale)

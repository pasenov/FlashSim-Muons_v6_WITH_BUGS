import os
import sys

import numpy as np

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from captum.attr import IntegratedGradients

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("Agg")

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from columns import eff_muon

def validation(validation_dataloader, model, device, tag):
    # Test the model

    model.eval()
    y_pred_list = []
    y_true_list = []
    y_pred_tag_list = []
    with torch.no_grad():
        for X, y in validation_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            y_pred = torch.sigmoid(y_pred)
            y_pred_list.append(y_pred.cpu().numpy())
            y_pred_tag = torch.round(y_pred)
            y_pred_tag_list.append(y_pred_tag.cpu().numpy())
            y_true_list.append(y.cpu().numpy())
            torch.cuda.empty_cache()

    y_pred_list = np.array(y_pred_list).flatten()
    y_true_list = np.array(y_true_list).flatten()
    y_pred_tag_list = np.array(y_pred_tag_list).flatten()

    # Check if output is a probability
    sum_pred = np.sum(y_pred_list)
    sum_true = np.sum(y_true_list)

    # save sum of predictions and true values to file
    with open(
        os.path.join(os.path.dirname(__file__), "figures", tag, "sum_pred_true.txt"),
        "w",
    ) as f:
        f.write(f"sum_pred: {sum_pred}\n")
        f.write(f"sum_true: {sum_true}\n")

    X, _ = validation_dataloader.dataset[:10000]
    X = X.to(device)

    ig = IntegratedGradients(model)
    attributions = ig.attribute(X, return_convergence_delta=False)
    attributions = attributions.cpu().numpy()

    # auc
    auc = roc_auc_score(y_true_list, y_pred_list)

    fpr, tpr, thresholds = roc_curve(y_true_list, y_pred_list)

    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], "k--")
    # auc on plot
    plt.text(0.5, 0.4, f"AUC: {auc:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "figures", tag, "roc_curve.pdf"),
        format="pdf",
    )
    plt.close()

    # histogram of predictions
    positive = y_pred_list[y_true_list == 1]
    negative = y_pred_list[y_true_list == 0]

    plt.figure(figsize=(10, 10))
    plt.title("Predictions")
    plt.hist(
        positive,
        bins=20,
        histtype="step",
        label="isReco",
        linewidth=2,
        edgecolor="b",
        fc=(0, 0, 1, 0.3),
        fill=True,
        range=(0, 1),
    )
    plt.hist(
        negative,
        bins=20,
        histtype="step",
        label="isNotReco",
        linewidth=2,
        edgecolor="r",
        fc=(1, 0, 0, 0.3),
        fill=True,
        range=(0, 1),
    )
    plt.yscale("log")
    plt.xlabel("Classifier output")
    plt.legend(frameon=False)
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "figures", tag, "predictions.pdf")
    )


def loss_plot(train_history, test_history, tag):
    plt.figure(figsize=(10, 10))
    plt.plot(train_history, label="Train loss")
    plt.plot(test_history, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(frameon=False)
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "figures", tag, "loss.pdf"),
        format="pdf",
    )
    plt.close()

def loss_plot_log(train_history, test_history, tag):
    plt.figure(figsize=(10, 10))
    plt.plot(train_history, label="Train loss")
    plt.plot(test_history, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "figures", tag, "loss_log.pdf"),
        format="pdf",
    )
    plt.close()


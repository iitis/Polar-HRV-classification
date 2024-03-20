import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from loading import load_fold_and_prepare_training_val_test_sets


def designate_bins(dataset):
    """
    Sturge's rule applied for *dataset*

    Arguments:
    ----------
      *dataset* (Numpy array) contains consecutive feature values

    Returns a Numpy array with includes left edges of bins and
    the right edge of the last bin.
    """
    no_of_bins = int(1 + np.ceil(np.log2(len(datasets["train"]))))
    interval = (np.max(dataset) - np.min(dataset)) / no_of_bins
    bins = np.arange(np.min(dataset), np.max(dataset), interval)
    return bins


def prepare_data_distribution(datasets, labels, path, single_folds=False):
    """
    Prepare histograms for all sets in *datasets*

    Arguments:
    ----------
      *datasets* (dictionary) contains consecutive sets for
                 distribution's plotting as Numpy arrays
      *labels* (dictionary) contains ground truth for
               all sets from *datasets* as Numpy arrays
      *path* (string) path for saving results
      *single_folds* (Boolean) defines whether for each iteration
                      a training, validation and test set should be
                      described (by default: False) or just single fold
                      per each iteration (True)
    """
    sns.set_style("whitegrid")
    os.makedirs(path, exist_ok=True)
    for dataset in list(datasets.keys()):
        values = {
            "treatment": datasets[dataset][
                np.argwhere(labels[dataset] == 1)
            ].flatten(),
            "control": datasets[dataset][
                np.argwhere(labels[dataset] == 0)
            ].flatten(),
        }
        adjustment = [["treatment", "red"], ["control", "green"]]
        fig, ax = plt.subplots(figsize=(4.5, 3))
        for group, color in adjustment:
            # Number of bins should be the same in the two compared groups.
            # Firstly, it is calculated on the basis of 'doane' method.
            if group == "treatment":
                bins = "doane"
            _, no_of_bins, _ = plt.hist(
                values[group],
                alpha=0.75,
                label=group,
                color=color,
                bins=bins,
                density=True,
            )
            bins = no_of_bins
        plt.xlabel("RR-value")
        plt.ylabel("Probability density")
        title = f", {dataset} set" if not single_folds else ""
        plt.title(f"Fold {fold_number + 1}{title}", fontsize=10)
        plt.legend()
        plt.tight_layout()
        if single_folds:
            name = f"{path}distribution_{fold_number + 1}"
        else:
            name = f"{path}distribution_{fold_number + 1}_{dataset}_set"
        plt.savefig(f"{name}.pdf", dpi=300)
        plt.close()


if __name__ == "__main__":
    timestep_length = 60  # 60 or 300
    data_location = (
        "./data/classification/equal_sizes/"
        f"{timestep_length}/"
        "individual_measurements_window_"
        f"{timestep_length}_5_folds.pkl"
    )
    path = f"./Results/distributions_{timestep_length}/"
    single_folds = False
    for fold_number in range(5):
        (
            X_train,
            y_train,
            _,
            X_validation,
            y_validation,
            _,
            X_test,
            y_test,
            _,
            data,
        ) = load_fold_and_prepare_training_val_test_sets(
            data_location, fold_number, standardize=False
        )
        if single_folds:
            datasets = {"test": X_test}
            labels = {"test": y_test}
        else:
            datasets = {
                "train": X_train,
                "validation": X_validation,
                "test": X_test,
            }
            labels = {
                "train": y_train,
                "validation": y_validation,
                "test": y_test,
            }
        # Sturge's rule applied for the train test
        # The number of windows should be similar in the two compared groups
        # but the length should be calculated based on one of the groups.
        # bins = designate_bins(datasets['train'])
        prepare_data_distribution(
            datasets, labels, path, single_folds=single_folds
        )

import numpy as np
from glob import glob
from data_analysis.utils_loading import load_results_file
from tsai.data.validation import combine_split_data
from tsai.data.core import TSDatasets, TSDataLoaders, TSTensor, TSLabelTensor
from tsai.data.transforms import Categorize


def prepare_standardization(X_train, X_test, X_validation=None):
    """
    Prepare data standardization for the flattened set, i.e.
    not for each sample/batch separately.

    Returns a list with preprocessed training, validation and test
    sets, separately.
    """
    X_train_flattened = X_train.flatten()
    mean, std = np.mean(X_train_flattened), np.std(X_train_flattened)
    if X_validation is not None:
        sets = [X_train, X_test, X_validation]
    else:
        sets = [X_train, X_test]
    data = map(lambda x: (x - mean) / std, sets)
    data = list(data).copy()
    return data


def extract_features_and_labels_from_data(dataframe):
    """
    Extract data features and corresponding labels
    from a given Pandas Dataframe
    """
    dataframe = dataframe.copy()
    dataframe["label"] = np.where(dataframe["group"] == "control", 0, 1)
    features = np.vstack(dataframe["RR_values"].values)
    labels = dataframe["label"].values
    dataframe["unique_id"] = (
        dataframe["group"] + "_" + dataframe["number"].astype(str)
    )
    unique_id = dataframe["unique_id"].values
    assert features.shape[0] == labels.shape[0]
    return features.astype("float32"), labels, unique_id


def select_data_for_fold(dataframe, fold_number, group):
    """
    Arguments:
    ----------
      *dataframe*: (Pandas Dataframe) contains data for all folds,
                   groups and training schemes
      *fold_number*: (int / string) fold for a current training
                     session
      *group*: (string) current training scheme: 'train', 'validation'
               or 'test'
    """
    current_data = dataframe.loc[dataframe[f"fold_{fold_number}"] == group][
        ["group", "number", "timestamps", "RR_values"]
    ]
    return current_data


def load_fold_and_prepare_training_val_test_sets(
    data_location, fold_number, standardize=False
):
    """
    Load raw data with all samples, extract windows related
    to the current fold and extract training, validation and test set
    with corresponding labels.

    Arguments:
    ----------
      *data_location* (string) location of the .pkl file with all samples
      *fold_number* (int) a number of the currently extracted fold
      *standardize* (Boolean, optional) defines whether data should
                    be standardized, by default: False

    Returns train, validation and test sets with labels.
      *X_{train, validation, test}*: (Numpy array) shape:
                                     (number of samples, number of features)
      *y_{train, validation, test}*: (Numpy array) shape: (number of samples, )
      *data*: (list) contains whole data for the selected fold, including
              anonimized ID of tested people
    """
    schemes = ["train", "validation", "test"]
    data_all_folds = load_results_file(data_location)
    data = map(
        lambda x: select_data_for_fold(data_all_folds, fold_number, x), schemes
    )
    data = list(data).copy()
    assert sum([len(data[i]) for i in range(len(data))]) == len(data_all_folds)
    extracted_data = map(
        lambda x: extract_features_and_labels_from_data(x), tuple(data)
    )
    extracted_data = list(extracted_data).copy()
    (
        (X_train, y_train, IDs_train),
        (X_validation, y_validation, IDs_validation),
        (X_test, y_test, IDs_test),
    ) = extracted_data

    if standardize:
        X_train, X_test, X_validation = prepare_standardization(
            X_train, X_test, X_validation
        )
    return (
        X_train,
        y_train,
        IDs_train,
        X_validation,
        y_validation,
        IDs_validation,
        X_test,
        y_test,
        IDs_test,
        data,
    )


def prepare_dataloaders(X, y, splits, parameters):
    """
    Prepare dataloaders for model training

    Arguments:
    ----------
      *X*: Numpy.ndarray of shapes: (number of samples, 1, length of a single
           time window) contains individual data samples
      *y*: Numpy.ndarray of shapes: (number of samples,) contains labels
           to the corresponding windows from *X*
      *splits*: tuple of fastcore.foundation.L objects containing samples
                from the training set at position 0 and samples from the
                test set at position 1
      *parameters*: dictionary containing following keys:
        -batch_size- int defining size of a single data batch
        -num_workers- int defining the number of CPU cores for parallel using
        -device- 'cuda' or None in case of CPU calculations

    Returns:
    --------
      *dataloaders* an object of tsai.data.core.TSDataLoaders class containing
                    training and validation datasets
    """
    assert X.shape[0] == y.shape[0]
    transforms = [None, Categorize()]
    datasets = TSDatasets(
        np.array(X).astype(np.float32),
        np.array(y).astype(np.int64),
        tfms=transforms,
        splits=splits,
        types=(TSTensor, TSLabelTensor),
        inplace=True,
    )
    dataloaders = TSDataLoaders.from_dsets(
        datasets.train,
        datasets.valid,
        shuffle_train=True,
        bs=parameters["batch_size"],
        num_workers=parameters["num_workers"],
        device=parameters["device"],
    )
    return dataloaders


def make_standardization_for_merged_training_val_sets(
    train_val_data_splits,
    X_train_val_data,
    y_train_val_data,
    X_test,
    IDs_train_val,
    verification=True,
):
    """
    Prepare data standardization in such a way. From merged train and
    validation sets extract the train set, adjust the standardization
    parameters and apply them to the validation and the test sets. Then,
    merge the training and the validation sets, but after this preprocessing.

    Arguments:
    ----------
       *train_val_data_splits* - (tuple of fastcore.foundation.L objects)
                                 contains indices of a training and
                                 a validation set
       *X_train_val_data* - (Numpy array) contains merged training and
                            validation data samples
       *y_train_val_data* - (Numpy array) contains labels for merged
                            training and validation samples
       *X_test* - (Numpy array) contains test samples
       *IDs_train_val* - (Numpy array) contains person names and their
                         positions in the merged training and validation
                         sets (e.g. control_2)
       *verification* - (Boolean) optional argument that defines whether
                        split verification should be performed (default: True)

    Returns:
    --------
       *X_train_val_data_S* - (Numpy array) contains merged training and
                              validation data samples after standardization
       *y_train_val_data_S* - (Numpy array) contains labels for merged
                              training and validation data samples after
                              standardization
       *train_val_data_splits_S* - (tuple of fastcore.foundation.L objects)
                                   contains indices of the training and
                                   validation sets after standardization
       *IDs_train_val_S* - (Numpy array) contains person names and their
                            positions in the merged training and validation
                            sets after standardization
       *X_test_S* - (Numpy array) test set after standardization
    """
    train_indices = list(train_val_data_splits[0])
    val_indices = list(train_val_data_splits[1])
    X_train, y_train, IDs_train = (
        X_train_val_data[train_indices],
        y_train_val_data[train_indices],
        IDs_train_val[train_indices],
    )
    X_validation, y_validation, IDs_validation = (
        X_train_val_data[val_indices],
        y_train_val_data[val_indices],
        IDs_train_val[val_indices],
    )
    X_train_S, X_test_S, X_validation_S = prepare_standardization(
        X_train, X_test, X_validation
    )
    X_train_val_data_S, y_train_val_data_S, train_val_data_splits_S = (
        combine_split_data([X_train_S, X_validation_S], [y_train, y_validation])
    )
    IDs_train_val_S = np.concatenate(
        (IDs_train_val[train_indices], IDs_train_val[val_indices])
    )

    if verification:
        unique_persons_train_val = np.unique(IDs_train_val)
        new_train_indices = list(train_val_data_splits_S[0])
        new_val_indices = list(train_val_data_splits_S[1])
        final_IDs_train = IDs_train_val_S[new_train_indices]
        final_y_train = y_train_val_data_S[new_train_indices]
        final_IDs_validation = IDs_train_val_S[new_val_indices]
        final_y_validation = y_train_val_data_S[new_val_indices]
        for person in unique_persons_train_val:
            # Count occurrence of a selected person in the first version of
            # the dataset and validate the labels
            initial_training_occurrence_of_given_person = np.argwhere(
                IDs_train == person
            )
            initial_validation_occurence_of_given_person = np.argwhere(
                IDs_validation == person
            )
            number_of_initial_training_occurrences = (
                initial_training_occurrence_of_given_person.shape[0]
            )
            number_of_initial_validation_occurrences = (
                initial_validation_occurence_of_given_person.shape[0]
            )
            initial_training_labels = np.sum(
                y_train[initial_training_occurrence_of_given_person]
            )
            initial_validation_labels = np.sum(
                y_validation[initial_validation_occurence_of_given_person]
            )
            ###
            # Check whether standardization changed the assignment of objects
            final_training_occurrence_of_given_person = np.argwhere(
                final_IDs_train == person
            )
            final_validation_occurrence_of_given_person = np.argwhere(
                final_IDs_validation == person
            )
            number_of_final_training_occurrences = (
                final_training_occurrence_of_given_person.shape[0]
            )
            number_of_final_validation_occurrences = (
                final_validation_occurrence_of_given_person.shape[0]
            )
            final_training_labels = np.sum(
                final_y_train[final_training_occurrence_of_given_person]
            )
            final_validation_labels = np.sum(
                final_y_validation[final_validation_occurrence_of_given_person]
            )
            assert (
                number_of_initial_training_occurrences
                == number_of_final_training_occurrences
            )
            assert (
                number_of_initial_validation_occurrences
                == number_of_final_validation_occurrences
            )
            assert initial_training_labels == final_training_labels
            assert initial_validation_labels == final_validation_labels

    return (
        X_train_val_data_S,
        y_train_val_data_S,
        train_val_data_splits_S,
        IDs_train_val_S,
        X_test_S,
    )


def find_path_to_selected_model(name, path_to_models="./"):
    """
    Find an exact path to the model with desired name's beginning.
    If there are more models in a single folder, an exception
    will be returned.

    Arguments:
    ----------
      *name* (string) defines a model's name beginning
      *path_to_models* (string, optional) path to the folder which will
                       be searched, by default it is a current folder

    Returns:
    --------
      An exact path with filename of the model found.
    """
    this_model_path = glob(
        f"{name}*.pkl", root_dir=path_to_models, recursive=False
    )
    if len(this_model_path) > 1:
        raise Exception(
            "There exists more than one model that fits the given criteria!"
        )
    elif len(this_model_path) == 0:
        raise Exception("No model following mentioned criteria!")
    this_model_path = f"{path_to_models}{this_model_path[0]}"
    return this_model_path


if __name__ == "__main__":
    data_location = (
        "./data/classification/60/"
        "individual_measurements_window_60_folds.pkl"
    )
    fold_number = 0
    (
        X_train,
        y_train,
        IDs_train,
        X_validation,
        y_validation,
        IDs_validation,
        X_test,
        y_test,
        IDs_test,
        data,
    ) = load_fold_and_prepare_training_val_test_sets(
        data_location, fold_number, standardize=True
    )

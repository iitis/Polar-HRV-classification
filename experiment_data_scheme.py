import os
import numpy as np
import pandas as pd
from glob import glob
from itertools import product
from sklearn.model_selection import StratifiedKFold
from data_analysis.utils_loading import (
    load_and_preprocess_data_for_single_person,
)
from data_analysis.utils_others import append_row_to_file


def extract_group_and_person_number(path):
    """
    Extract information about the current person
    from the selected path.

    Argument:
    ---------
      *path* (str) contains group ID ('control' or 'treatment')
             as well as the number of the person in this group

    Returns a dictionary with two keys: 'group' and 'number'.
    """
    try_control = path.find("control_")
    if try_control != -1:
        person_info = {"group": "control"}
        if path.find(".csv") == -1:
            person_info["number"] = path[(try_control + 8) :]
        else:
            person_info["number"] = path[(try_control + 8) : path.find(".csv")]
        return person_info
    else:
        try_treatment = path.find("treatment_")
        person_info = {"group": "treatment"}
        if path.find(".csv") == -1:
            person_info["number"] = path[(try_treatment + 10) :]
        else:
            person_info["number"] = path[
                (try_treatment + 10) : path.find(".csv")
            ]
        return person_info


def assign_label_to_current_person(cur_person, X_train, X_val, X_test):
    """
    Checks in which set a currently selected person is assigned.

    Arguments:
      *cur_person* (dictionary or string):
         - A dictionary contains two keys: 'group' and 'number'
         - A string has the final form, e.g. 'control_1', 'treatment_1'
      *X_train* (Numpy array) contains string IDs from persons from the train
                set, e.g. 'control_1', 'treatment_1'
      *X_val* (Numpy array) contains string IDs from persons from
              the validation set, e.g. 'control_1', 'treatment_1'
      *X_test* (Numpy array) contains string IDs from persons from the test
               set, e.g. 'control_1', 'treatment_1'

    Returns a string 'train', 'validation' or 'test'.
    """
    if isinstance(cur_person, dict):
        name = f'{cur_person["group"]}_{cur_person["number"]}'
    elif isinstance(cur_person, str):
        name = cur_person
    else:
        raise ValueError("A type of the argument cur_person is wrong!")

    if name in X_train:
        return "train"
    elif name in X_val:
        return "validation"
    elif name in X_test:
        return "test"
    else:
        return ValueError(
            "Current person is not present in the training/"
            "validation/test set"
        )


def save_current_fold_info_to_file(
    X, X_train, X_val, X_test, path="./data/classification/", name="fold"
):
    """
    Save information about current fold to the file.
    For each person save his/her group and set (train, validation
    or test).

    Arguments:
    ----------
      *X*: (Numpy array) contains all persons taking part
           in a given measurements.
      *X_train* (Numpy array) contains string IDs from persons from the train
                set, e.g. 'control_1', 'treatment_1'
      *X_val* (Numpy array) contains string IDs from persons from
              the validation set, e.g. 'control_1', 'treatment_1'
      *X_test* (Numpy array) contains string IDs from persons from the test
               set, e.g. 'control_1', 'treatment_1'
      *path* (optional string) path in which a file with results will be saved
      *name* (optional string) name of the file with results
    """
    os.makedirs(path, exist_ok=True)
    append_row_to_file(f"{path}{name}.csv", "person_group_name;set")
    X = np.sort(X)
    for person in X:
        person_set = assign_label_to_current_person(
            str(person), X_train, X_val, X_test
        )
        append_row_to_file(f"{path}{name}.csv", f"{person};{person_set}")


def collect_all_persons(files):
    """
    Collect all persons taking part in the measurement.
    Argument:
    ---------
      *files* (list) contains paths to files from consecutive persons

    Returns:
    --------
      *X* (Numpy array) contains consecutive persons as strings, e.g.
          'control_1', 'treatment_1', etc.
      *y* (Numpy array) contains classes for corresponding persons
          from X, 1 for treatment and 0 for control persons
    """
    X, y = [], []
    for cur_file in files:
        group_and_person_number = extract_group_and_person_number(cur_file)
        # 1 for treatment, 0 for control persons
        X.append(
            f"{group_and_person_number['group']}_"
            f"{group_and_person_number['number']}"
        )
        if group_and_person_number["group"] == "control":
            y.append(0)
        elif group_and_person_number["group"] == "treatment":
            y.append(1)
        else:
            raise NameError("Wrong name of the desired group!")
    return np.array(X), np.array(y)


def load_data_for_individuals(files, parameters):
    """
    Load RR intervals data, divide them into the time windows and prepare
    Pandas Dataframe with individual measurements.

    Arguments:
    ----------
      *files* (list) contains paths to files from consecutive persons
      *parameters* (dictionary) contains parameters of the experiment

    Returns:
    --------
      Pandas Dataframe with individual windows. It contains the following
      columns: 'group', 'number', 'timestamps' and 'RR_values'.
    """
    # Load data from consecutive persons
    individual_windows = []
    for cur_file in files:
        group_and_person_number = extract_group_and_person_number(cur_file)
        print(f"Preparing of {group_and_person_number}")
        # 1 for treatment, 0 for control persons
        data = load_and_preprocess_data_for_single_person(
            parameters,
            cur_person_group=group_and_person_number["group"],
            cur_person_number=group_and_person_number["number"],
        )
        windows = [
            *data.rolling(window=parameters["window_size"], method="table")
        ]
        # Remove windows having less elements than 'window_size'
        filtered_windows = [
            *filter(
                lambda windows: len(windows) == parameters["window_size"],
                windows,
            )
        ]
        for window in filtered_windows:
            assert (
                window["Phone timestamp"].values.shape[0]
                == parameters["window_size"]
            )
            assert (
                window["RR-interval [ms]"].values.shape[0]
                == parameters["window_size"]
            )
            individual_windows.append(
                [
                    group_and_person_number["group"],
                    group_and_person_number["number"],
                    window["Phone timestamp"].values,
                    window["RR-interval [ms]"].values,
                ]
            )
    individual_measurements_divided = pd.DataFrame(
        individual_windows,
        columns=["group", "number", "timestamps", "RR_values"],
    )
    return individual_measurements_divided


def verify_correctness_of_split(
    X_train, y_train, X_val, y_val, X_test, y_test, no_of_folds
):
    # 60 / N samples in the test set, 30 / N for each class
    no_of_persons_in_test_val_set = int(60 / no_of_folds)
    assert np.sum(y_test) == int(no_of_persons_in_test_val_set / 2)
    assert y_test.shape[0] == no_of_persons_in_test_val_set
    assert np.sum(y_val) == int(no_of_persons_in_test_val_set / 2)
    assert y_val.shape[0] == no_of_persons_in_test_val_set
    no_of_persons_in_training_set = int(60 - 2 * no_of_persons_in_test_val_set)
    assert np.sum(y_train) == int(no_of_persons_in_training_set / 2)
    assert y_train.shape[0] == no_of_persons_in_training_set
    X_concatenated = np.concatenate((X_train, X_val, X_test), axis=0)
    # There should be 60 different elements in X
    assert np.unique(X_concatenated).shape[0] == 60
    y_concatenated = np.concatenate((y_train, y_val, y_test), axis=0)
    # Only two classes should be available
    assert np.unique(y_concatenated).shape[0] == 2
    # There should be 30 treatments and 30 control persons
    assert np.sum(y_concatenated) == 30
    assert y_concatenated.shape[0] == 60


def main(files, parameters):
    """
    Main function preparing the following steps:
    1) Collect all persons taking part in the measurement.
    2) Prepare N splits with mutually exclusive sets of persons.
    3) For each split prepare a training, a validation and a test set.
    4) Save data into the pickle file.

    Returns Pandas Dataframe containing individual windows and such
    columns as 'group', 'number' 'fold_0', ..., 'fold_{N-1}' (where N
    is the number of folds), 'timestamps' and 'RR_values'.
    """
    X, y = collect_all_persons(files)
    individual_windows = load_data_for_individuals(files, parameters)
    # Divide the collected data into some splits
    skfolds = StratifiedKFold(
        n_splits=parameters["no_of_folds"],
        random_state=parameters["random_state"],
        shuffle=True,
    )
    joint_setup = [*product(skfolds.split(X, y))].copy()
    for i, (train_and_val_idx, test_idx) in enumerate(skfolds.split(X, y)):
        print(f"Split no: {i}")
        individual_windows.insert(2 + i, f"fold_{i}", 0)
        if i == (parameters["no_of_folds"] - 1):
            val_idx = joint_setup[0][0][1]
        else:
            val_idx = joint_setup[i + 1][0][1]
        # Remove validation indices from the training set
        train_idx = np.array([i for i in train_and_val_idx if i not in val_idx])
        X_test, y_test = X[test_idx], y[test_idx]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        verify_correctness_of_split(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            parameters["no_of_folds"],
        )

        save_current_fold_info_to_file(
            X,
            X_train,
            X_val,
            X_test,
            parameters["results_path"],
            name=f"fold_{i}",
        )

        for group, current_set in zip(
            ["train", "validation", "test"], [X_train, X_val, X_test]
        ):
            for idx in range(current_set.shape[0]):
                info = extract_group_and_person_number(current_set[idx])
                individual_windows.loc[
                    (individual_windows["group"] == info["group"])
                    & (individual_windows["number"] == info["number"]),
                    f"fold_{i}",
                ] = group

    os.makedirs(parameters["results_path"], exist_ok=True)
    individual_windows["number"] = individual_windows["number"].astype(int)
    individual_windows.to_pickle(
        f'{parameters["results_path"]}individual_measurements_'
        f'window_{parameters["window_size"]}_'
        f'{parameters["no_of_folds"]}_folds.pkl'
    )
    print("Saving completed.")
    return individual_windows


if __name__ == "__main__":
    path = (
        "./data/Exp_1_HRV_calculations_anonimized_raw_data/"
    )
    filetypes = ["control_*.csv", "treatment_*.csv"]
    files = []
    for filetype in filetypes:
        files.extend(glob(f"{path}{filetype}"))

    parameters = {
        "sequence_range": "windows",
        "method": "RMSSD",
        "no_of_folds": 5,
        "random_state": 8,
        "adjacent_beats_for_removing": "5 seconds",
        "threshold_for_hole_duration": "30 seconds",
        "time_after_hole_for_removing": "15 seconds",
        "cut_time_from_start": "45 seconds",
        "cut_time_before_finish": "45 seconds",
        "window_size": 60,  # 60 or 300
        "main_folder": path,
        "interpolation": False,
        "plot": False,
    }
    parameters["results_path"] = (
        f"./data/classification/CV_{parameters['no_of_folds']}/redesigned/"
        f"{parameters['window_size']}/"
    )
    individual_windows = main(files, parameters)

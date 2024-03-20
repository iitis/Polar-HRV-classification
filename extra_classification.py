from tsai.data.validation import get_splits
import torch
import os
import numpy as np

from datetime import datetime
from loading import (
    extract_features_and_labels_from_data,
    make_standardization_for_merged_training_val_sets,
    prepare_dataloaders,
)
from data_analysis.utils_loading import load_results_file
from data_analysis.utils_others import append_row_to_file
from basic_classification import (
    pipeline_train_evaluate_model,
    set_seed,
)
from hyperparameters import set_hyperparameters


def perform_leave_one_out_CV_experiments(hyperparameters, ID_of_start_person=0):
    """
    Perform Leave One Out Cross-Validation experiments with a full dataset.
    It means that 60 persons in the dataset will be divided into 60 folds
    and 60 separate training sessions will be performed. Each time, in the
    training and validation set will be windows from 59 persons while in
    the test set will be windows from the remaining person.

    Arguments:
    ----------
      *hyperparameters* - (dict) contains experiment's hyperparameters
      *ID_of_start_person* - (optional int) selects ID of the first person
                             in the loop; in such a case experiments may be
                             interrupted and there is no need to start from
                             the first fold, default: 0; it cannot be greater
                             than ID of the last person (59)
    """
    data_location = hyperparameters["data_location"]
    os.makedirs(hyperparameters["saving_folder"], exist_ok=True)

    data_all_folds = load_results_file(data_location)
    data_all_folds.drop(
        columns=["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"], inplace=True
    )
    unique_persons = (
        data_all_folds.groupby(["group", "number"]).size().index.values
    )
    # Totally there are 60 folds. We do not want to store in memory 60 huge
    # datasets, therefore we are selecting only the current setup.
    assert ID_of_start_person < unique_persons.shape[0]
    for current_fold in range(ID_of_start_person, unique_persons.shape[0]):
        # Seed should be set here to ensure that we can start training
        # from different persons
        set_seed(hyperparameters["seed"])
        data_all_folds_for_calc = data_all_folds.copy()
        assert current_fold >= 0 and current_fold < 60
        hyperparameters["fold_number"] = current_fold
        test_person = unique_persons[current_fold]
        test_group, test_number = test_person[0], test_person[1]
        test_data = data_all_folds_for_calc.loc[
            (data_all_folds_for_calc["group"] == test_group)
            & (data_all_folds_for_calc["number"] == test_number)
        ].copy()
        data_all_folds_for_calc.drop(test_data.index, inplace=True)
        train_data = data_all_folds_for_calc.reset_index(drop=True)
        test_data.reset_index(drop=True, inplace=True)

        X_train_val_data, y_train_val_data, IDs_train_val = (
            extract_features_and_labels_from_data(train_data)
        )
        X_test, y_test, IDs_test = extract_features_and_labels_from_data(
            test_data
        )
        train_val_data_splits = get_splits(
            y_train_val_data,
            n_splits=1,
            valid_size=hyperparameters["validation_size_LOOCV"],
            test_size=0.0,
            stratify=True,
            balance=False,
            shuffle=True,
            random_state=hyperparameters["seed"],
            show_plot=False,
            verbose=True,
        )

        if hyperparameters["data_standardization"]:
            (
                X_train_val_data,
                y_train_val_data,
                train_val_data_splits,
                IDs_train_val,
                X_test,
            ) = make_standardization_for_merged_training_val_sets(
                train_val_data_splits,
                X_train_val_data,
                y_train_val_data,
                X_test,
                IDs_train_val,
                verification=True,
            )

        y_train_val_data, y_test = list(
            map(lambda x: x.astype(np.int64), [y_train_val_data, y_test])
        )
        X_train_val_data, X_test = list(
            map(lambda x: x.astype(np.float32), [X_train_val_data, X_test])
        )
        val_indices = list(train_val_data_splits[1])
        X_validation, y_validation, IDs_validation = (
            X_train_val_data[val_indices],
            y_train_val_data[val_indices],
            IDs_train_val[val_indices],
        )

        device = "cuda" if torch.cuda.is_available() else None
        dataloaders = prepare_dataloaders(
            X_train_val_data,
            y_train_val_data,
            train_val_data_splits,
            {
                "batch_size": hyperparameters["batch_size"],
                "num_workers": 0,
                "device": device,
            },
        )
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_for_saving_model = (
            f'LOOCV_{hyperparameters["model"]}_fold_{current_fold}_'
            f'max_{hyperparameters["number_of_epochs"]}_epochs_'
            f'lr_{hyperparameters["learning_rate"]}_'
            f'seed_{hyperparameters["seed"]}_{current_time}'
        )

        full_path_for_saving_individual_results = (
            f'./{hyperparameters["saving_folder"]}/'
            f'{hyperparameters["saving_filename"]}_'
            f'LOOCV_fold_{current_fold}_seed_{hyperparameters["seed"]}_'
            f"individual_preds"
        )

        pipeline_train_evaluate_model(
            dataloaders,
            X_validation,
            y_validation,
            IDs_validation,
            X_test,
            y_test,
            IDs_test,
            hyperparameters,
            full_path_for_saving_individual_results,
            train_model_from_scratch=hyperparameters[
                "train_model_from_scratch"
            ],
            path_of_loaded_model=hyperparameters["model_exact_loading_path"],
            name_for_saving_model=name_for_saving_model,
        )


if __name__ == "__main__":
    # For experiment with Leave One Out Cross-Validation we only use 'GRU_FCN'
    model = "GRU_FCN"
    timestep_length = 60  # 60 or 300

    validation_size_LOOCV = 0.2

    global_hyperparameters = set_hyperparameters(
        model, timestep_length, part="single_training"
    )
    global_hyperparameters["data_location"] = (
        "./data/classification/equal_sizes/"
        f'{global_hyperparameters["timestep_length"]}/'
        "individual_measurements_window_"
        f'{global_hyperparameters["timestep_length"]}_5_folds.pkl'
    )
    global_hyperparameters["saving_folder"] = (
        f"Results/fixed_LOOCV_{model}_timestep_{timestep_length}_part_2"
    )
    os.makedirs(global_hyperparameters["saving_folder"], exist_ok=True)
    global_hyperparameters["saving_filename"] = (
        f"summary_results_LOOCV_{model}_timestep_{timestep_length}"
    )
    model_header = (
        "RNN_hidden_size;RNN_no_layers;RNN_dropout;" "RNN_bidirectional"
    )
    main_header = (
        "time_length;no_of_fold;model;model_id;seed;number_of_epochs;"
        "learning_rate;batch_size;early_stop_patience;lr_patience;"
        f"{model_header};validation_window_accuracy;validation_person_accuracy;"
        "voting_threshold;test_window_accuracy;test_person_accuracy;"
    )
    append_row_to_file(
        f'./{global_hyperparameters["saving_folder"]}/'
        f'{global_hyperparameters["saving_filename"]}.csv',
        main_header,
    )

    # Prepare a final set of dictionary for further calculations
    basic_keys = [
        "saving_folder",
        "saving_filename",
        "validation_size_for_final_training",
        "data_standardization",
        "data_location",
        "timestep_length",
        "number_of_features",
        "number_of_classes",
        "early_stopping_patience",
        "learning_rate_patience",
        "model",
    ]
    hyperparameters = {key: global_hyperparameters[key] for key in basic_keys}
    hyperparameters["batch_size"] = global_hyperparameters["batch_sizes"][0]
    hyperparameters["learning_rate"] = global_hyperparameters["learning_rates"][
        0
    ]
    hyperparameters["RNN_hidden_size"] = global_hyperparameters[
        "RNN_hidden_sizes"
    ][0]
    hyperparameters["RNN_no_of_layers"] = global_hyperparameters[
        "RNN_numbers_of_layers"
    ][0]
    hyperparameters["RNN_dropout"] = global_hyperparameters["RNN_dropouts"][0]
    hyperparameters["RNN_bidirectional"] = global_hyperparameters[
        "RNN_bidirectional_opts"
    ][0]
    hyperparameters["validation_size_LOOCV"] = validation_size_LOOCV
    hyperparameters["number_of_epochs"] = 100
    hyperparameters["model_id"] = None
    hyperparameters["train_model_from_scratch"] = True
    hyperparameters["model_exact_loading_path"] = None
    hyperparameters["path_to_more_models"] = None

    for seed in global_hyperparameters["seeds"]:
        hyperparameters["seed"] = seed
        ID_of_start_person = 0
        perform_leave_one_out_CV_experiments(
            hyperparameters, ID_of_start_person=ID_of_start_person
        )

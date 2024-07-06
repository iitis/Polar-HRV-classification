import os
from copy import deepcopy
import random
import torch
import numpy as np
import pandas as pd
from tsai.data.validation import combine_split_data, get_splits
from tsai.inference import load_learner
from tsai.models.TransformerModel import TransformerModel
from tsai.models.RNN_FCN import GRU_FCN
from tsai.learner import Learner, load_all
from tsai.metrics import accuracy
from fastai.callback.tracker import (
    EarlyStoppingCallback,
    SaveModelCallback,
    ReduceLROnPlateau,
    CSVLogger,
)
from sklearn.metrics import accuracy_score
from loading import (
    find_path_to_selected_model,
    load_fold_and_prepare_training_val_test_sets,
    prepare_dataloaders,
)
from data_analysis.utils_others import append_row_to_file
from hyperparameters import set_hyperparameters
from evaluation import (
    assess_individual_persons_based_on_multiple_windows,
    get_current_prediction,
    select_optimal_threshold,
)
from datetime import datetime
from pandas.testing import assert_frame_equal
from itertools import product


def set_seed(value):
    """
    Set deterministic results according to the given value
    (including random, numpy and torch libraries)
    """
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results_to_file(hyperparameters, results):
    """
    Save a single set of results to the file

    Arguments:
      *hyperparameters* (dictionary) contains the following keys:
       -transformer_dimension-, -transformer_number_of_heads-,
       -transformer_feedforward_dimension-, -transformer_enc_layers-,
       -transformer_dropout-, -RNN_hidden_size-, -RNN_no_of_layers-,
       -RNN_dropout-, -RNN_bidirectional-, -timestep_length-,
       -fold_number-, -model-, -model_id-, -seed-, -number_of_epochs-,
       -learning_rate-, -batch_size-, -early_stopping_patience-,
       -learning_rate_patience-, -saving_folder-, -saving_filename-
      *results* (dictionary) contains the following keys:
       -validation_window_accuracy-, -validation_person_accuracy-,
       -test_window_accuracy-, -test_person_accuracy-
    """
    if hyperparameters["model"] == "transformer":
        model_information = (
            f"{hyperparameters['transformer_dimension']};"
            f"{hyperparameters['transformer_number_of_heads']};"
            f"{hyperparameters['transformer_feedforward_dimension']};"
            f"{hyperparameters['transformer_enc_layers']};"
            f"{hyperparameters['transformer_dropout']};"
        )
    elif hyperparameters["model"] == "GRU_FCN":
        model_information = (
            f"{hyperparameters['RNN_hidden_size']};"
            f"{hyperparameters['RNN_no_of_layers']};"
            f"{hyperparameters['RNN_dropout']};"
            f"{hyperparameters['RNN_bidirectional']};"
        )
    else:
        raise ValueError("This model is not implemented!")

    row_with_results = (
        f"{hyperparameters['timestep_length']};"
        f"{hyperparameters['fold_number']};"
        f"{hyperparameters['model']};"
        f"{hyperparameters['model_id']};"
        f"{hyperparameters['seed']};"
        f"{hyperparameters['number_of_epochs']};"
        f"{hyperparameters['learning_rate']};"
        f"{hyperparameters['batch_size']};"
        f"{hyperparameters['early_stopping_patience']};"
        f"{hyperparameters['learning_rate_patience']};"
        f"{model_information}"
        f"{results['validation_window_accuracy']};"
        f"{results['validation_person_accuracy']};"
        f"{results['voting_threshold']};"
        f"{results['test_window_accuracy']};"
        f"{results['test_person_accuracy']};"
    )
    append_row_to_file(
        f'./{hyperparameters["saving_folder"]}/'
        f'{hyperparameters["saving_filename"]}.csv',
        row_with_results,
    )


def prepare_model(parameters):
    """
    A dictionary containing experiment's hyperparameters:
      -number_of_features- (int) number of features in the dataset,
                           for univariate TS is equal to 1
      -number_of_classes- (int) for treatment/healthy control set to 2
      -timestep_length- (int) length of the individual sequence
      PARAMETERS FOR TRANSFORMER:
      -transformer_dimension- (int)
      -transformer_number_of_heads- (int)
      -transformer_feedforward_dimension- (int)
      -transformer_enc_layers- (int)
      -transformer_dropout- (float)
      PARAMETERS FOR GRU_FCN:
      -RNN_hidden_size- (int)
      -RNN_no_of_layers- (int)
      -RNN_dropout- (float)
      -RNN_bidirectional- (Boolean)

    Returns tsAI model prepared according to the given hyperparameters.
    """
    if parameters["model"] == "transformer":
        model = TransformerModel(
            c_in=parameters["number_of_features"],
            c_out=parameters["number_of_classes"],
            # According to tsai documentation: "number of features
            # created by the model"
            # d_model usually between 128 and 1024
            d_model=parameters["transformer_dimension"],
            # the number of parallel attention heads, usually 8-16
            n_head=parameters["transformer_number_of_heads"],
            # Dimension of the feedforward network
            # usually between 256 and 4096
            d_ffn=parameters["transformer_feedforward_dimension"],
            # "Number of sub-encoder layers"
            # usually between 2 and 8
            n_layers=parameters["transformer_enc_layers"],
            # By default 0.1 in tsAI
            dropout=parameters["transformer_dropout"],
        )
        # rest of the hyperparameters will be default

    elif parameters["model"] == "GRU_FCN":
        model = GRU_FCN(
            c_in=parameters["number_of_features"],
            c_out=parameters["number_of_classes"],
            seq_len=parameters["timestep_length"],
            # By default 100 in tsAI
            hidden_size=parameters["RNN_hidden_size"],
            # Number of RNN layers, by default 1 in tsAI
            rnn_layers=parameters["RNN_no_of_layers"],
            bias=True,
            # By default is set to 0.8 in tsAI
            rnn_dropout=parameters["RNN_dropout"],
            bidirectional=parameters["RNN_bidirectional"],
        )
        # rest of the hyperparameters will be default
    return model


def model_inference(
    X_test,
    y_test,
    name_of_test_set,
    batch_size,
    learner=None,
    loading_path=None,
    X_validation=None,
    y_validation=None,
    name_of_validation_set=None,
):
    """
    Test a given model on the test (and optionally also validation) set.
    If learner is not given as an argument it has to be loaded from the file.

    Arguments:
    ----------
      *X_test* - (Numpy array) features of the test set, dimensions:
                 (number of samples, number of time steps)
      *y_test* - (Numpy array) labels for *X_test*, dimensions:
                 (number of samples,)
      *name_of_test_set* - (string) defines name for the above set for inference
      *batch_size* - (int) size of a single batch with samples
      *learner* - (fastai.learner.Learner or None, optional) Learner with model
                  for inference; if None then *path* should be given
      *path* - (string, optional) path for the saved learner
      *X_validation* - (Numpy array, optional) features of the validation set,
                       dimensions: (number of validation samples, number of
                       time steps)
      *y_validation* - (Numpy array, optional) labels for *X_validation*,
                       dimensions: (number of validation samples,)
      *name_of_validation_set* - (optional string) defines name for the validation
                                 set for inference

    Returns:
    --------
    A dictionary with test_probabilities, test_labels, test_predictions
    and test_accuracy and potentially validation_probabilities,
    validation_labels, validation_predictions and validation_accuracy.
    """
    assert learner is not None or loading_path is not None
    assert (
        X_validation is not None
        and y_validation is not None
        and name_of_validation_set is not None
    ) or (X_validation is None and y_validation is None)

    if learner is None:
        cpu = False if torch.cuda.is_available() else True
        learner = load_learner(
            f"{loading_path.replace('.pth', '')}.pth", cpu=cpu
        )
    results = {}

    to_inference = [[name_of_test_set, X_test, y_test]]
    if X_validation is not None:
        to_inference.append(
            [name_of_validation_set, X_validation, y_validation]
        )

    for name, features, labels in to_inference:
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=1)
        (
            results[f"{name}_probabilities"],
            results[f"{name}_labels"],
            results[f"{name}_predictions"],
        ) = learner.get_X_preds(
            features.astype(np.float32), labels.astype(np.int64), bs=batch_size
        )
        results[f"{name}_predictions"] = (
            results[f"{name}_predictions"].replace(" ", "")[1:-1].split(",")
        )
        results[f"{name}_predictions"] = [
            int(label) for label in results[f"{name}_predictions"]
        ]
        results[f"{name}_accuracy"] = accuracy_score(
            labels, results[f"{name}_predictions"]
        )
        print(f'Accuracy for the {name} set: {results[f"{name}_accuracy"]}.')
    return results


def prepare_majority_voting_for_fold_predictions(
    model_predictions,
    ground_truth,
    persons_IDs_in_fold,
    filename=None,
    saving_folder="./",
    save_dataframe=True,
):
    """
    Perform majority voting for consecutive persons of the fold.

    Arguments:
    ----------
      *model_predictions* - (Numpy array) a 1D array containing either 0's
                             or 1's regarding predictions for consecutive
                             persons
      *ground_truth* - (Numpy array) a 1D array containing either 0's or 1's
                       containing true values for consecutive persons
      *persons_IDs_in_fold* - (Numpy array) a 1D array containing names of
                              consecutive persons
      *filename* - (optional string) name of file for saving, default None
      *saving_folder* - (optional string) path for saving results
      *save_dataframe* - (optional Boolean) defines whether a dataframe will
                         be saved to the file or not; default: True

    Returns:
    --------
      *dataframe* - (Pandas Dataframe) contains results for consecutive persons
                    of the corresponding fold
      *fold_accuracy* - (float) accuracy for the whole fold according to
                        the majority voting rule
    """
    assert (
        persons_IDs_in_fold.shape[0]
        == model_predictions.shape[0]
        == ground_truth.shape[0]
    )
    unique_persons = np.unique(persons_IDs_in_fold)
    # A list for storing results for consecutive persons in a given fold
    final_partial_results, detailed_partial_results, partial_GT = [], [], []
    for person in unique_persons:
        selected_indices = np.argwhere(persons_IDs_in_fold == person)
        subset_predictions = model_predictions[selected_indices].flatten()
        individual_result = get_current_prediction(subset_predictions)
        if "control" in person:
            partial_GT.append(0)
        elif "treatment" in person:
            partial_GT.append(1)
        else:
            raise ValueError(
                "A person is either in the control or in the treatment group!"
            )
        final_partial_results.append(individual_result["prediction"])
        detailed_partial_results.append(individual_result["disease_percentage"])
    final_partial_results = np.array(final_partial_results)
    partial_GT = np.array(partial_GT)
    fold_accuracy = (
        np.sum(final_partial_results == partial_GT) / partial_GT.shape[0]
    )
    # Create Pandas Dataframe based on stored results
    zipped_results = list(
        zip(unique_persons, final_partial_results, detailed_partial_results)
    )
    dataframe = pd.DataFrame(
        zipped_results, columns=["name", "prediction", "disease_percentage"]
    )
    if save_dataframe:
        if filename is None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fold_results_{current_time}.csv"
        dataframe.to_csv(f"{saving_folder}/{filename}.csv", sep=";")
    return dataframe, fold_accuracy


def test_prepare_majority_voting_for_fold_predictions():
    """
    Unittest of prepare_majority_voting_for_fold_predictions()
    """
    test_persons = np.array(
        [
            "control_3",
            "control_3",
            "control_3",
            "control_5",
            "control_5",
            "control_5",
            "control_5",
            "treatment_1",
            "treatment_1",
            "treatment_1",
            "treatment_5",
            "treatment_5",
            "treatment_5",
            "control_8",
            "control_8",
            "control_8",
            "control_8",
            "control_8",
        ]
    )
    test_ground_truth = np.array(
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    )
    test_model_predictions = np.array(
        [0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]
    )
    test_dataframe, test_accuracy = (
        prepare_majority_voting_for_fold_predictions(
            test_model_predictions,
            test_ground_truth,
            test_persons,
            save_dataframe=False,
        )
    )
    test_dataframe_ground_truth = [
        ["control_3", 0, 1 / 3],
        ["control_5", 0, 0.5],
        ["treatment_1", 1, 2 / 3],
        ["treatment_5", 1, 1],
        ["control_8", 1, 0.6],
    ]
    test_gt_dataframe = pd.DataFrame(
        test_dataframe_ground_truth,
        columns=["name", "prediction", "disease_percentage"],
    )
    test_gt_dataframe.sort_values(by="name", inplace=True)
    test_dataframe.reset_index(inplace=True)
    assert_frame_equal(test_dataframe, test_gt_dataframe)
    gt_accuracy = 0.8
    assert np.abs(gt_accuracy - test_accuracy) >= 1e-6


def train_model(hyperparameters, dataloaders, path):
    """
    Train single model in tsAI.

    Arguments:
    ----------
       *hyperparameters* - (dictionary) stores following keys: 'learning_rate',
                           'number_of_epochs', 'early_stopping_patience',
                           'learning_rate_patience', 'saving_folder'
       *dataloaders* - (tsai.data.core.TSDataLoaders) contains train and
                       validation datasets
       *path* - (string) for saving results

    Returns:
    --------
       *learn* - (fastai.learner.Learner) learned model with intrinsic
                 properties
    """
    model = prepare_model(hyperparameters)
    learn = Learner(
        dataloaders,
        model,
        lr=hyperparameters["learning_rate"],
        metrics=accuracy,
    )
    learn.fit(
        n_epoch=hyperparameters["number_of_epochs"],
        cbs=[
            EarlyStoppingCallback(
                # Decision regarding early stopping will be taken
                # according to the values of the validation loss
                monitor="valid_loss",
                patience=hyperparameters["early_stopping_patience"],
            ),
            SaveModelCallback(
                # Save best model according to the validation loss
                # not necessarily the model from the last epoch
                monitor="valid_loss",
                # Models are saved in a subfolder './models'
                fname=path,
                at_end=False,
                with_opt=True,
            ),
            ReduceLROnPlateau(
                # Learning rate will be reduced about one order of magnitude
                # after each 'patience' steps without validation loss
                # improvements
                monitor="valid_loss",
                patience=hyperparameters["learning_rate_patience"],
            ),
            CSVLogger(
                # Save results after consecutive epochs to the .csv file
                fname=f'./{hyperparameters["saving_folder"]}/{path}.csv',
                append=True,
            ),
        ],
    )
    # Export saves best model taking into account SaveModelCallback
    # learn.save_all(f"./models/{path}", verbose=True)
    learn.export(f"./models/{path}.pth")
    return learn


def train_model_with_merged_training_validation_folds(hyperparameters):
    """
    In the grid search experiments, networks are trained using training
    folds while are validated with validation folds. In this function,
    training and validation folds are merged, a percentage of the total
    samples is separated and a network is trained with a bigger training
    set, but still without using test folds.

    Arguments:
    ----------
      *hyperparameters* - (dictionary) stores all necessary hyperparameters
    """
    os.makedirs(hyperparameters["saving_folder"], exist_ok=True)
    set_seed(hyperparameters["seed"])
    for fold_number in range(5):
        print(f"Calculations for fold: {fold_number}")
        hyperparameters["fold_number"] = fold_number
        hyperparameters["model_id"] = None
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
            _,
        ) = load_fold_and_prepare_training_val_test_sets(
            hyperparameters["data_location"], fold_number, standardize=True
        )
        X1, y1, _ = combine_split_data(
            [X_train, X_validation], [y_train, y_validation]
        )
        retrain_splits = get_splits(
            y1,
            n_splits=1,
            valid_size=hyperparameters["validation_size_for_final_training"],
            test_size=0.0,
            stratify=True,
            balance=False,
            shuffle=True,
            random_state=hyperparameters["seed"],
            show_plot=False,
            verbose=True,
        )
        # Sanity check
        assert len(retrain_splits[0]) + len(retrain_splits[1]) == y1.shape[0]
        assert np.isclose(
            len(retrain_splits[1])
            / (len(retrain_splits[0]) + len(retrain_splits[1])),
            hyperparameters["validation_size_for_final_training"],
            rtol=1e-03,
        )
        ##############
        device = "cuda" if torch.cuda.is_available() else None
        dataloaders = prepare_dataloaders(
            X1,
            y1,
            retrain_splits,
            {
                "batch_size": hyperparameters["batch_size"],
                "num_workers": 0,
                "device": device,
            },
        )
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_model_name = (
            f'retrained_final_model_{hyperparameters["model"]}_'
            f'fold_{fold_number}_max_{hyperparameters["number_of_epochs"]}_'
            f'epochs_lr_{hyperparameters["learning_rate"]}_'
            f'seed_{hyperparameters["seed"]}'
        )
        full_path_for_saving_individual_results = (
            f'./{hyperparameters["saving_folder"]}/'
            f'{hyperparameters["saving_filename"]}_'
            f'fold_{fold_number}_seed_{hyperparameters["seed"]}_'
            f"individual_preds"
        )
        if hyperparameters["train_model_from_scratch"]:
            name_for_saving_model = (
                f"best_model_{base_model_name}_{current_time}"
            )
            path_of_model_for_loading = None
        else:
            name_for_saving_model = None
            if hyperparameters["model_exact_loading_path"] is not None:
                path_of_model_for_loading = hyperparameters[
                    "model_exact_loading_path"
                ]
            else:
                name_for_loaded_model = f"best_model_{base_model_name}_"
                path_of_model_for_loading = find_path_to_selected_model(
                    name=name_for_loaded_model,
                    path_to_models=hyperparameters["path_to_more_models"],
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
            path_of_loaded_model=path_of_model_for_loading,
            name_for_saving_model=name_for_saving_model,
        )


def pipeline_train_evaluate_model(
    dataloaders,
    X_validation,
    y_validation,
    IDs_validation,
    X_test,
    y_test,
    IDs_test,
    hyperparameters,
    full_path_for_saving_individual_results,
    train_model_from_scratch=True,
    path_of_loaded_model=None,
    name_for_saving_model=None,
):
    """
    Perform model training using previously prepared dataloaders and
    evaluation with a test, and potentially also a validation dataset.

    Arguments:
    ----------
       *dataloaders* an object of tsai.data.core.TSDataLoaders class containing
                     training and validation datasets
       *X_validation* (Numpy array) features of the validation set,
                      dimensions: (number of validation samples, number of time
                      steps)
       *y_validation* (Numpy array) labels for *X_validation*,
                      dimensions: (number of validation samples,)
       *IDs_validation* (Numpy array) contains person names and their positions
                  in the validation set (e.g. treatment_3)
       *X_test* (Numpy array) features of the test set, dimensions:
                (number of samples, number of time steps)
       *y_test* (Numpy array) labels for *X_test*, dimensions:
                (number of samples,)
       *IDs_test* (Numpy array) contains person names and their positions
                  in the test set (e.g. treatment_3)
       *hyperparameters* (dict) contains the following keys:
           :::transformer:::
           -transformer_dimension-, -transformer_number_of_heads-,
           -transformer_feedforward_dimension-, -transformer_enc_layers-,
           -transformer_dropout-
           :::GRU_FCN:::
           -RNN_hidden_size-, -RNN_no_of_layers- -RNN_dropout-,
           -RNN_bidirectional-
        -timestep_length-, -fold_number-, -model-, -model_id-,
        -seed-, -number_of_epochs-, -learning_rate-, -batch_size-,
        -early_stopping_patience-, -learning_rate_patience-, -saving_folder-,
        -saving_filename-
       *full_path_for_saving_individual_results* (string) full path, including
                      folder and filename, for saving .npz file with results
                      for individual windows
       *train_model_from_scratch* (Boolean, optional) defines whether a model
                                  should be trained for scratch (default)
                                  or should be loaded from file
       *path_of_loaded_model* (string, optional) contains name of the file
                               with model for loading, important only when
                               *train_model_from_scratch* is False
       *name_for_saving_model* (string, optional) contains name of the file
                               with model for saving, important only when
                               *train_model_from_scratch* is True
    """
    assert train_model_from_scratch or (
        not train_model_from_scratch and path_of_loaded_model is not None
    )

    cpu = False if torch.cuda.is_available() else True
    if train_model_from_scratch:
        learn = train_model(hyperparameters, dataloaders, name_for_saving_model)
        evaluate_learner = load_learner(
            f'./models/{name_for_saving_model}.pth',
            cpu=cpu)
    else:
        evaluate_learner = load_learner(
            f"{path_of_loaded_model}", cpu=cpu)
        # evaluate_learner = load_all(
        # f"{path_of_loaded_model}", verbose=True)
    # Prepare a full prediction using the validation set
    validation_fold_results = model_inference(
        X_validation,
        y_validation,
        "validation",
        batch_size=1,
        # hyperparameters["batch_size"],
        learner=evaluate_learner,
    )
    validation_predictions = np.array(
        validation_fold_results["validation_predictions"]
    )
    individual_persons_validation_results, individuals_validation_GTs = (
        assess_individual_persons_based_on_multiple_windows(
            validation_predictions,
            y_validation,
            IDs_validation,
        )
    )
    final_threshold, validation_person_accuracy, thresholding_results = (
        select_optimal_threshold(
            individual_persons_validation_results, individuals_validation_GTs
        )
    )
    # Now, prepare a prediction using the test set and the optimal threshold
    test_fold_results = model_inference(
        X_test,
        y_test,
        "test",
        batch_size=1,
        # hyperparameters["batch_size"],
        learner=evaluate_learner,
    )
    test_individual_predictions = np.array(
        test_fold_results["test_predictions"]
    )
    test_total_predictions, test_individuals_GTs = (
        assess_individual_persons_based_on_multiple_windows(
            test_individual_predictions, y_test, IDs_test
        )
    )
    individual_person_thresholding = np.where(
        test_total_predictions > (final_threshold / 100), 1, 0
    )
    final_test_accuracy = (
        np.count_nonzero(individual_person_thresholding == test_individuals_GTs)
        / test_individuals_GTs.shape[0]
    )

    np.savez_compressed(
        full_path_for_saving_individual_results,
        validation_predictions=validation_predictions,
        validation_gt=y_validation,
        validation_person_IDs=IDs_validation,
        test_predictions=test_individual_predictions,
        test_gt=y_test,
        test_person_IDs=IDs_test,
    )
    print(f"Accuracy for the current fold: {final_test_accuracy}.")
    results = {
        "validation_window_accuracy": validation_fold_results[
            "validation_accuracy"
        ],
        "validation_person_accuracy": validation_person_accuracy,
        "voting_threshold": final_threshold,
        "test_window_accuracy": test_fold_results["test_accuracy"],
        "test_person_accuracy": final_test_accuracy,
    }
    save_results_to_file(
        hyperparameters,
        results,
    )


def single_grid_search_model_training(global_hyperparameters, model_id):
    """
    Prepare a network training for a single set of
    hyperparameters (as a part of the grid search experiment)
    and save results to the .csv file.

    Arguments:
    ----------
      *global_hyperparameters*: (dictionary) contains following experiment
            hyperparameters: *saving_folder* (str), *saving_filename* (str),
            *validation_size_for_final_training* (float),
            *number_of_epochs* (int), *data_standardization* (Boolean),
            *timestep_length* (int), *seed* (int), *batch_size* (int),
            *learning_rate* (float), *number_of_features* (int),
            *number_of_classes* (int), *early_stopping_patience* (int),
            *learning_rate_patience* (int), *model* (str),
            *transformer_dimension* (int), *transformer_number_of_heads* (int),
            *transformer_feedforward_dimension* (int),
            *transformer_enc_layers* (int) *transformer_dropout* (float),
            *RNN_hidden_size* (int), *RNN_no_of_layers* (int),
            *RNN_dropout* (float), *RNN_bidirectional* (Boolean)
      *model_id*: (int) ID of the current training iteration; it will be
                  included in the saving path
    """
    set_seed(global_hyperparameters["seed"])
    hyperparameters = deepcopy(global_hyperparameters)
    for fold_number in range(5):
        hyperparameters["fold_number"] = fold_number
        hyperparameters["model_id"] = model_id
        print(f"Calculations for fold: {fold_number}")
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
            _,
        ) = load_fold_and_prepare_training_val_test_sets(
            hyperparameters["data_location"],
            fold_number,
            standardize=hyperparameters["data_standardization"],
        )
        X1, y1, splits1 = combine_split_data(
            [X_train, X_validation], [y_train, y_validation]
        )
        device = "cuda" if torch.cuda.is_available() else None
        dataloaders = prepare_dataloaders(
            X1,
            y1,
            splits1,
            {
                "batch_size": hyperparameters["batch_size"],
                "num_workers": 0,
                "device": device,
            },
        )
        path = (
            f'{hyperparameters["model"]}_model_id_{model_id}_'
            f'{hyperparameters["model"]}_fold_{fold_number}_'
            f'max_{hyperparameters["number_of_epochs"]}_'
            f'epochs_lr_{hyperparameters["learning_rate"]}'
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
            hyperparameters["saving_folder"],
            train_model_from_scratch=False,
            path_of_loaded_model=f"models/best_model_{path}.pkl",
            name_for_saving_model=None,
        )
        # After training and inference a single model would not be useful
        if os.path.exists(f"models/best_model_{path}.pkl"):
            os.remove(f"models/best_model_{path}.pkl")
        if os.path.exists(f"models/{path}.pkl"):
            os.remove(f"models/{path}.pkl")


if __name__ == "__main__":
    model = "GRU_FCN"  # "transformer" or "GRU_FCN"
    timestep_length = 60  # 60 or 300
    mode = "single_training"
    ###################################################
    # Options for *mode*:
    #  -single_training- training of models with a single set of hyperparams
    #  -grid- performing grid search, i.e. training of models with multiple
    #         sets of hyperparameters
    #  -evaluation- evaluation of trained (and saved) models for different
    #               threshold values
    ###################################################
    global_hyperparameters = set_hyperparameters(
        model, timestep_length, part=mode
    )
    global_hyperparameters["data_location"] = (
        "./data/classification/equal_sizes/"
        f'{global_hyperparameters["timestep_length"]}/'
        "individual_measurements_window_"
        f'{global_hyperparameters["timestep_length"]}_5_folds.pkl'
    )
    global_hyperparameters["train_model_from_scratch"] = True

    # In the case of 'header' modification it is necessary to update
    # 'save_results_state' function.
    if global_hyperparameters["model"] == "transformer":
        model_header = (
            "transf_dim;transf_no_heads;transf_feedforward;"
            "transf_enc_layers;transf_dropout"
        )
    elif global_hyperparameters["model"] == "GRU_FCN":
        model_header = (
            "RNN_hidden_size;RNN_no_layers;RNN_dropout;" "RNN_bidirectional"
        )
    else:
        raise ValueError("This model is not implemented!")
    header = (
        "time_length;no_of_fold;model;model_id;seed;number_of_epochs;"
        "learning_rate;batch_size;early_stop_patience;lr_patience;"
        f"{model_header};validation_window_accuracy;validation_person_accuracy;"
        "voting_threshold;test_window_accuracy;test_person_accuracy;"
    )
    append_row_to_file(
        f'./{global_hyperparameters["saving_folder"]}/'
        f'{global_hyperparameters["saving_filename"]}.csv',
        header,
    )

    # Prepare a final set of dictionary for further calculations
    basic_keys = [
        "saving_folder",
        "saving_filename",
        "validation_size_for_final_training",
        "number_of_epochs",
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
    if mode in ["single_training", "grid"]:
        hyperparameters["train_model_from_scratch"] = True
        hyperparameters["path_to_more_models"] = None
    elif mode == "evaluation":
        hyperparameters["train_model_from_scratch"] = False
        main_path_to_models = "./models/"
        number_of_folds = 5
    hyperparameters["model_exact_loading_path"] = None

    if global_hyperparameters["model"] == "transformer":
        settings = product(
            global_hyperparameters["seeds"],
            global_hyperparameters["batch_sizes"],
            global_hyperparameters["learning_rates"],
            global_hyperparameters["transformer_dimensions"],
            global_hyperparameters["transformer_numbers_of_heads"],
            global_hyperparameters["transformer_feedforward_dimensions"],
            global_hyperparameters["transformer_numbers_enc_layers"],
            global_hyperparameters["transformer_dropouts"],
        )
        model_id = 0
        for setting in settings:
            hyperparameters["seed"] = setting[0]
            hyperparameters["batch_size"] = setting[1]
            hyperparameters["learning_rate"] = setting[2]
            hyperparameters["transformer_dimension"] = setting[3]
            hyperparameters["transformer_number_of_heads"] = setting[4]
            hyperparameters["transformer_feedforward_dimension"] = setting[5]
            hyperparameters["transformer_enc_layers"] = setting[6]
            hyperparameters["transformer_dropout"] = setting[7]
            if mode == "single_training":
                train_model_with_merged_training_validation_folds(
                    hyperparameters
                )
            elif mode == "grid":
                single_grid_search_model_training(hyperparameters, model_id)
                model_id += 1
            elif mode == "evaluation":
                hyperparameters["path_to_more_models"] = (
                    f"{main_path_to_models}{model}_{timestep_length}/"
                )
                train_model_with_merged_training_validation_folds(
                    hyperparameters
                )

    elif global_hyperparameters["model"] == "GRU_FCN":
        settings = product(
            global_hyperparameters["seeds"],
            global_hyperparameters["batch_sizes"],
            global_hyperparameters["learning_rates"],
            global_hyperparameters["RNN_hidden_sizes"],
            global_hyperparameters["RNN_numbers_of_layers"],
            global_hyperparameters["RNN_dropouts"],
            global_hyperparameters["RNN_bidirectional_opts"],
        )
        model_id = 0
        for setting in settings:
            hyperparameters["seed"] = setting[0]
            hyperparameters["batch_size"] = setting[1]
            hyperparameters["learning_rate"] = setting[2]
            hyperparameters["RNN_hidden_size"] = setting[3]
            hyperparameters["RNN_no_of_layers"] = setting[4]
            hyperparameters["RNN_dropout"] = setting[5]
            hyperparameters["RNN_bidirectional"] = setting[6]
            if mode == "single_training":
                train_model_with_merged_training_validation_folds(
                    hyperparameters
                )
            elif mode == "grid":
                single_grid_search_model_training(hyperparameters, model_id)
                model_id += 1
            elif mode == "evaluation":
                hyperparameters["path_to_more_models"] = (
                    f"{main_path_to_models}{model}_{timestep_length}/"
                )
                train_model_with_merged_training_validation_folds(
                    hyperparameters
                )

    else:
        raise ValueError("This model is not implemented!")

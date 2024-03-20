from numpy.testing import assert_array_almost_equal
from tsai.inference import load_learner
from loading import load_fold_and_prepare_training_val_test_sets
from data_analysis.utils_loading import load_results_file
import numpy as np
import pandas as pd
import pickle


def get_current_prediction(windows_predictions, threshold=0.5):
    """
    Prepares a majority voting for a subset of predictions.

    Arguments:
    ----------
      *windows_predictions* (1D Numpy array) contains predictions
                            for the selected subset of data
      *threshold* (float) a value from the range (0, 1) defining
                  the threshold value; above this value a label 'treatment'
                  will be assigned

    Returns a dictionary with keys:
     -'disease_percentage': the ratio of the disease's predictions
     -'prediction': 1 if 'disease_percentage' is above the threshold,
                    0 in the opposite case
    """
    no_of_disease_predictions = np.sum(windows_predictions)
    disease_percentage = (
        no_of_disease_predictions / windows_predictions.shape[0]
    )
    result = {"disease_percentage": disease_percentage}
    result["prediction"] = 1 if disease_percentage > threshold else 0
    return result


def evaluate_single_set_of_predictions(
    dataframe, predictions, ground_truth, threshold
):
    """
    Evaluate a single set of the algorithm's predictions
    (i.e., for a selected fold)

    Arguments:
    ----------
      *dataframe* (Pandas Dataframe) contains columns: 'group',
                  'number' and corresponding windows (e.g. 'RR_values')
      *predictions* (Numpy array) contains 0's or 1's as the algorithm's
                    predictions
      *ground_truth* (Numpy array) contains 0's or 1's as ground truth
                     for all windows
      *threshold* (float) a value from the range (0, 1) defining
                  the threshold value; above this value a label 'treatment'
                  will be assigned

    Returns:
    --------
      *results* (list of lists) contains individual results; the order of
                elements is as follows: group, person, prediction (0 or 1),
                percentage of disease predictions
    """
    assert np.array_equal(ground_truth, dataframe["label"].values)
    # Get IDs of people from the test set and the number of samples
    consecutive_individuals = dataframe.groupby(["group", "number"]).size()
    # An array just for the verification
    presence_array = np.zeros(ground_truth.shape[0])
    results = []
    for group, number in consecutive_individuals.index.values:
        current_elements = dataframe.loc[
            (dataframe["group"] == group) & (dataframe["number"] == number)
        ].index.values
        presence_array[current_elements] += 1
        current_predictions = predictions[current_elements]
        voting_result = get_current_prediction(current_predictions, threshold)
        results.append(
            [
                group,
                number,
                voting_result["prediction"],
                voting_result["disease_percentage"],
            ]
        )
    # Each window should be present only once
    assert np.sum(presence_array) == presence_array.shape[0]
    assert np.min(presence_array) == np.max(presence_array) == 1.0
    return results


def main_evaluation(
    folds_list, all_data_location, batch_size, threshold, evaluated_set="test"
):
    results = []
    for fold_number, fold_path in enumerate(folds_list):
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
            all_data_location, fold_number, standardize=True
        )
        if evaluated_set == "train":
            samples, ground_truth = X_train, y_train
            index = 0
        elif evaluated_set == "validation":
            samples, ground_truth = X_validation, y_validation
            index = 1
        elif evaluated_set == "test":
            samples, ground_truth = X_test, y_test
            index = 2
        else:
            raise ValueError("Wrong evaluation option!")
        learner = load_learner(fold_path)
        samples = np.expand_dims(samples, axis=1)
        test_probas, test_labels, test_predictions = learner.get_X_preds(
            samples, ground_truth, bs=batch_size
        )
        test_predictions = test_predictions.replace(" ", "")[1:-1].split(",")
        predictions = np.array([int(label) for label in test_predictions])

        dataframe = data[index].copy().reset_index()
        results.extend(
            evaluate_single_set_of_predictions(
                dataframe, predictions, ground_truth, threshold
            )
        )
    return results


def calculate_overall_accuracy(dataframe):
    """
    Calculate overall accuracy based on Pandas Dataframe
    with results for individuals

    Arguments:
    ----------
      *dataframe* (Pandas DataFrame) contains following columns:
                  'group', 'number', 'prediction' and
                  'ratio_of_treatment_windows'

    Returns a float value corresponding to the overall accuracy.
    """
    dataframe["prediction"] = dataframe["prediction"].astype("int")
    no_of_correct_classifications = 0
    for index, row in dataframe.iterrows():
        if row["group"] == "control":
            gt_class = 0
        elif row["group"] == "treatment":
            gt_class = 1
        else:
            raise ValueError("Wrong value in the column: group!")
        if gt_class == row["prediction"]:
            no_of_correct_classifications += 1
    overall_accuracy = no_of_correct_classifications / len(dataframe)
    return overall_accuracy


def select_best_hyperparams_from_grid_search(
    path, filename, aggregated_param, model
):
    """
    Select the best set of hyperparameters based on grid search
    results. Save a dataframe with mean results sorted according
    to *aggregated_param*.

    Arguments:
    ----------
      *path* (str) path to the folder with results
      *filename* (str) name of the file with results for all folds
                 and all tested sets of hyperparams
      *aggregated_param*: (str) name of the column that will be used
                          for the selection of the best model
      *model*: (str) name of the tested model: 'transformer' or 'GRU_FCN'

    Returns:
    --------
      A dictionary with the best hyperparameters.
    """
    assert model in ["transformer", "GRU_FCN"]
    dataframe = pd.read_csv(f"{path}{filename}", sep=";")
    dataframe = dataframe.dropna(axis=1, how="all")
    general_params = [
        "learning_rate",
        "batch_size",
        "no_of_fold",
    ]
    GRU_FCN_params = [
        "RNN_hidden_size",
        "RNN_no_layers",
        "RNN_dropout",
        "RNN_bidirectional",
    ]
    transformer_params = [
        "transf_dim",
        "transf_no_heads",
        "transf_feedforward",
        "transf_enc_layers",
        "transf_dropout",
    ]
    if model == "transformer":
        additional_params = transformer_params
    elif model == "GRU_FCN":
        additional_params = GRU_FCN_params
    set_of_unique_params = general_params + additional_params
    set_of_unique_params.remove("no_of_fold")
    # First, calculate mean for each fold with different seeds.
    # In the second step, calculate mean for consecutive folds.
    mean_results = (
        dataframe.groupby(general_params + additional_params)
        .mean()
        .groupby(level=set_of_unique_params)
        .mean()
    )
    mean_results = mean_results.sort_values(
        by=[aggregated_param], ascending=False
    )
    mean_results.to_csv(f"{path}mean_{filename}", sep=";")
    # Unittest: some random results were verified with manual
    # calculations and their results were exactly the same
    # like returned by this method.
    titles = list(mean_results.head(1).index.names)
    values = list(mean_results.head(1).index.tolist()[0])
    best_hyperparams = {titles[i]: values[i] for i in range(len(values))}
    return best_hyperparams


def assess_individual_persons_based_on_multiple_windows(
    model_predictions,
    y_validation,
    IDs_validation,
):
    """
    Assess predictions of a given model and evaluate which percentage
    of individual time windows was classified as disease.

    Arguments:
    ----------
        *model_predictions*: (Numpy array) contains model predictions, in terms
                             of 0's and 1's for individual windows;
        *y_validation*: (Numpy array) contains 0's and 1's for individual
                        windows; ground truth for individual windows
        *IDs_validation*: (Numpy array) contains information about persons
                          related to the consecutive windows from *y_validation*
                          as strings, e.g. 'control_1', 'treatment_1', etc.

    Returns:
    --------
        *individual_persons_results*: (Numpy array) contains floats between 0
                                      and 1 and describes which percentage
                                      of windows for individual persons was
                                      classified as disease
        *individual_GTs*: (Numpy array) contains 0's and 1's defining true
                          class of consecutive persons
    """
    assert (
        y_validation.shape[0]
        == model_predictions.shape[0]
        == IDs_validation.shape[0]
    )
    unique_persons_validation = np.unique(IDs_validation)
    individual_persons_results, individuals_GTs = [], []
    for person in unique_persons_validation:
        indices = np.argwhere(IDs_validation == person)
        ground_truth = np.unique(y_validation[indices].flatten())
        assert len(ground_truth) == 1
        individuals_GTs.append(ground_truth[0])
        predictions = model_predictions[indices].flatten()
        # Measure how many windows were indicated as "disease"
        disease_windows = np.count_nonzero(predictions == 1)
        disease_windows_percentage = disease_windows / predictions.shape[0]
        individual_persons_results.append(disease_windows_percentage)
    individual_persons_results = np.array(individual_persons_results)
    individuals_GTs = np.array(individuals_GTs)
    return individual_persons_results, individuals_GTs


def select_optimal_threshold(individual_persons_results, individuals_GTs):
    """
    Evaluate person windows for different threshold values and select
    the threshold giving the highest accuracy

    Arguments:
    ----------
       *individual_persons_results*: (Numpy array) contains floats between 0
                                      and 1 and describes which percentage
                                      of windows for individual persons was
                                      classified as disease
       *individual_GTs*: (Numpy array) contains 0's and 1's defining true
                          class of consecutive persons

    Returns:
    --------
       *final_threshold*: (int) defines the optimal threshold
       *selected_accuracy*: (float) defines person's classification accuracy
                            for *final_threshold*
       *thresholding_results*: (dict) contains accuracy for consecutive
                                threshold values
    """
    thresholding_results = {}
    # Evaluate consecutive persons for different thresholds
    for threshold in range(1, 100, 1):
        individual_person_thresholding = np.where(
            individual_persons_results > (threshold / 100), 1, 0
        )
        this_threshold_accuracy = (
            np.count_nonzero(individual_person_thresholding == individuals_GTs)
            / individuals_GTs.shape[0]
        )
        thresholding_results[threshold] = this_threshold_accuracy

    # Select threshold ensuring the highest accuracy
    maximum_accuracies = [
        int(k)
        for k, v in thresholding_results.items()
        if v == max(thresholding_results.values())
    ]
    # If multiple windows achieved the same result, select majority
    # voting or median threshold.
    if len(maximum_accuracies) > 1:
        if 50 in maximum_accuracies:
            final_threshold = 50
        else:
            final_threshold = int(np.median(maximum_accuracies))
    else:
        final_threshold = maximum_accuracies[0]
    selected_accuracy = thresholding_results[final_threshold]
    return final_threshold, selected_accuracy, thresholding_results


def test_evaluate_single_set_of_predictions():
    """
    Unittest of evaluate_single_set_of_predictions()
    """
    # Dataframe has columns: 'group' ('treatment' or 'control'), 'number'
    # (potentially also RR_values) and label
    test_dataframe = [
        ['control', 5, [1046, 951, 931], 0],
        ['control', 5, [886, 923, 862], 0],
        ['control', 5, [914, 955, 970], 0],
        ['control', 5, [973, 1009, 1100], 0],
        ['control', 5, [1044, 1103, 1058], 0],
        ['control', 15, [1253, 1211, 1219], 0],
        ['control', 15, [1264, 1114, 1293], 0],
        ['control', 15, [1205, 1290, 1288], 0],
        ['control', 15, [1368, 1406, 1287], 0],
        ['control', 15, [1314, 1230, 1163], 0],
        ['control', 15, [1333, 1312, 1374], 0],
        ['treatment', 1, [1313, 1244, 1282], 1],
        ['treatment', 1, [1228, 1332, 1257], 1],
        ['treatment', 1, [1320, 1284, 1336], 1],
        ['treatment', 1, [1310, 1223, 1191], 1],
        ['treatment', 1, [1229, 1165, 1304], 1],
        ['treatment', 1, [1209, 1123, 1139], 1],
        ['treatment', 3, [951, 931, 891], 1],
        ['treatment', 3, [888, 912, 886], 1],
        ['treatment', 3, [923, 862, 914], 1],
        ['treatment', 3, [955, 970, 973], 1],
        ['treatment', 3, [1009, 1100, 1044], 1],
        ['treatment', 3, [1103, 1058, 1253], 1],
        ['treatment', 3, [1211, 1219, 1264], 1],
        ['treatment', 3, [1114, 1293, 1205], 1],
        ['treatment', 3, [1290, 1288, 1368], 1],
    ]
    test_dataframe = pd.DataFrame(
        test_dataframe, columns=["group", "number", "RR_values", "label"]
    )
    predictions = np.array([0, 0, 0, 1, 1,
                            0, 0, 1, 1, 1, 0,
                            1, 1, 1, 1, 1, 1,
                            0, 1, 1, 1, 0, 0, 0, 0, 0])
    ground_truth = np.array([0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1])
    threshold = 0.5
    gt_result = pd.DataFrame([
        ['control', 5, 0, 0.4],
        ['control', 15, 0, 0.5],
        ['treatment', 1, 1, 1.],
        ['treatment', 3, 0, 3 / 9]
    ])
    output_result = pd.DataFrame(
        evaluate_single_set_of_predictions(
            test_dataframe, predictions, ground_truth, threshold
        )
    )
    assert gt_result.equals(output_result)


def test_assess_individual_persons_based_on_multiple_windows():
    """
    Unittest of assess_individual_persons_based_on_multiple_windows()
    """
    test_model_prediction = np.array([
        1, 1, 0, 1, 0,
        0, 1, 0, 1, 0,
        0, 1, 0, 1, 1,
        1, 0, 1, 0, 0,
        1, 1, 0, 1, 0
    ])
    test_y_validation = np.array([
        0, 1, 0, 1, 0,
        1, 1, 1, 0, 1,
        0, 0, 1, 1, 1,
        1, 0, 1, 1, 0,
        1, 1, 1, 0, 1
    ])
    test_IDs_validation = np.array([
        'control_1', 'treatment_1', 'control_1', 'treatment_2', 'control_1',
        'treatment_1', 'treatment_1', 'treatment_2', 'control_1', 'treatment_2',
        'control_1', 'control_1', 'treatment_1', 'treatment_1', 'treatment_1',
        'treatment_2', 'control_1', 'treatment_1', 'treatment_1', 'control_1',
        'treatment_1', 'treatment_2', 'treatment_2', 'control_1', 'treatment_1'
    ])
    test_individuals_GTs = np.array(
        [0, 1, 1]
    )
    # Number of 1's:
    # 'control_1': 4 / 9,
    # 'treatment_1': 6 / 10
    # 'treatment_2': 3 / 6
    test_individual_persons_results = np.array([
        4 / 9, 6 / 10, 3 / 6
    ])
    res_individual_persons_results, res_individuals_GTs = (
        assess_individual_persons_based_on_multiple_windows(
            test_model_prediction, test_y_validation, test_IDs_validation
        )
    )
    assert_array_almost_equal(test_individuals_GTs, res_individuals_GTs)
    assert_array_almost_equal(
        test_individual_persons_results, res_individual_persons_results
    )


def test_select_optimal_threshold():
    """
    Unittest of test_select_optimal_threshold()
    """
    # TEST 1)
    test_individual_persons_results = np.array([
        0.061, 0.0158, 0.512, 0.53, 0.991
    ])
    test_individual_GTs = np.array([
        0, 1, 0, 0, 1
    ])
    res_threshold, res_accuracy, res_thresholding = select_optimal_threshold(
        test_individual_persons_results, test_individual_GTs
    )
    gt_thresholding = {
        1: 0.4, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2, 6: 0.2,
        52: 0.6,
    }
    for i in range(7, 52):
        gt_thresholding[i] = 0.4
    for i in range(53, 100):
        gt_thresholding[i] = 0.8
    gt_threshold = 76
    gt_accuracy = 0.8
    assert res_threshold == gt_threshold
    assert res_accuracy == gt_accuracy
    assert res_thresholding == gt_thresholding
    # TEST 2)
    test_individual_persons_results = np.array([
        0.061, 0.0158, 0.467, 0.475, 0.991
    ])
    gt_thresholding[47] = 0.6
    for i in range(48, 100):
        gt_thresholding[i] = 0.8
    gt_threshold = 50
    res_threshold, res_accuracy, res_thresholding = select_optimal_threshold(
        test_individual_persons_results, test_individual_GTs
    )
    assert res_threshold == gt_threshold
    assert res_accuracy == gt_accuracy
    assert res_thresholding == gt_thresholding


if __name__ == "__main__":
    test_evaluate_single_set_of_predictions()
    test_assess_individual_persons_based_on_multiple_windows()
    all_data_location = (
        "./data/classification/60/"
        "individual_measurements_window_60_folds.pkl"
    )
    folds_list = []
    for fold_no in range(5):
        folds_list.append(
            f"./Results/test_model_fold_{fold_no}_"
            "30_epochs_constant_lr_0.0001.pkl"
        )
    batch_size = 64
    threshold = 0.5
    evaluation_result = {}
    for set_name in ["train", "validation", "test"]:
        print(f"Calculations for {set_name} set.")
        evaluation_result[set_name] = main_evaluation(
            folds_list,
            all_data_location,
            batch_size,
            threshold,
            evaluated_set=set_name,
        )

    with open("./Results/results_summary.pkl", "wb") as file_saving:
        pickle.dump(evaluation_result, file_saving)

    summary_of_results = load_results_file("./Results/results_summary.pkl")
    for set_name in ["train", "validation", "test"]:
        individual_results = pd.DataFrame(
            summary_of_results[set_name],
            columns=[
                "group",
                "number",
                "prediction",
                "ratio_of_treatment_windows",
            ],
        )
        accuracy = calculate_overall_accuracy(individual_results)
        print(f"Overall accuracy for the {set_name} set: {accuracy}.")

    # Select best hyperparameters
    main_path = "./Results/"
    aggregated_param = "validation_window_accuracy"
    scenario_paths = [
        f"{main_path}grid_search_GRU_FCN_part_0/",
        f"{main_path}grid_search_GRU_FCN_part_1/",
        f"{main_path}grid_search_transformer_part_0/",
    ]
    scenario_filenames = [
        "summary_results_GRU_FCN_part_0.csv",
        "summary_results_GRU_FCN_part_1.csv",
        "summary_results_transformer_part_0.csv",
    ]
    scenario_models = ["GRU_FCN", "GRU_FCN", "transformer"]
    best_hyperparams = []

    for path, filename, model in zip(
        scenario_paths, scenario_filenames, scenario_models
    ):
        best_hyperparams.append(
            select_best_hyperparams_from_grid_search(
                path, filename, aggregated_param, model
            )
        )

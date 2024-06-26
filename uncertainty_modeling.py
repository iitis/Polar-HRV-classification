from itertools import product
from datetime import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def display_p_values(p_value):
    """
    Prepares a string with properly formatted p-value.

    Argument:
    ---------
       *p_value* (float) represents p-value

    Returns:
    --------
       String for displaying obtained p-value in plots.
    """
    if p_value > 0.02:
        present_p_value = f"$P={p_value:.2f}$"
    elif p_value >= 0.002 and p_value <= 0.02:
        present_p_value = f"$P={p_value:.3f}$"
    elif p_value >= 0.001 and p_value < 0.002:
        present_p_value = f"$P={p_value:.4f}$"
    elif p_value < 0.001:
        present_p_value = "$P<0.001$"
    elif np.isnan(p_value):
        present_p_value = "P is undefined"
    return present_p_value


def moving_window_variances(table, M):
    """
    Calculate variances in moving windows containing M consecutive elements.

    Arguments:
    ----------
        *table* (numpy.ndarray): Input array of N elements.
        *M* (int): Number of consecutive elements in each window.

    Returns:
    --------
        *numpy.ndarray*: Array of variances for each moving window.
    """
    N = len(table)
    if M > N:
        raise ValueError("Window size M must be less than or equal to the "
                         "length of the input array.")

    variances = np.array([np.var(table[i:i + M], ddof=0) for i in range(N - M + 1)])
    # The value located at i-th position corresponds to the variance of elements
    # from the i-th to the (i+M)-th index.
    return variances


def designate_prediction_based_on_minimum_variance(predictions, variances, M):
    """
    Select a window having the least variance and get a prediction
    corresponding to such a window.

    Arguments:
    ----------
        *predictions* (numpy.ndarray): Input array of consecutive predictions
        *variances* (numpy.ndarray): Input array of prediction variances
                                     corresponding to subsequent windows.
        *M* (int): Number of consecutive elements in each window.
    """
    assert len(predictions) == (len(variances) + M - 1)
    minimum_variance = np.min(variances)
    indices_of_minimum_variance = np.argwhere(variances == minimum_variance)
    # If multiple windows have the same variance get the first window
    final_prediction = predictions[indices_of_minimum_variance[0][0]]
    return final_prediction


def plot_combined_variance_RR_predictions_for_individual(
    mean_RR_values_for_selected_individual,
    local_variances,
    evaluation,
    parameters,
    title_prefix=None
):
    """
    Prepare a plot with combined results of analysis. 
    The first subplot presents R-R interval data values in consecutive
    timesteps.
    The second subplot presents rolling variances based on subsequent
    time windows.
    The third subplot presents prediction of the loaded method.
    Green lines correspond to correct predictions, red lines are related
    to the opposite case.

    Arguments:
    ----------
       *mean_RR_values_for_selected_individual* (numpy.ndarray) mean values
                         of R-R intervals from consecutive time windows:
                         a single value represents a single window
       *local_variances* (numpy.ndarray) represents the variances of
                         predictions within consecutive time windows
       *evaluation* (numpy.ndarray) 1's correspond to correct predictions, 0's
                    denote wrong predictions; one prediction per single window
       *parameters* a dictionary containing the following keys:
         -time_window_range- (int) represents M consecutive windows for
                             variance calculation
         -group_of_individual- (str) represents name of the group of the
                               selected person ('control' or 'treatment')
         -ID_of_individual- (int) ID of the selected person without its group
         -statistics- (float) represents Pearson's r
         -p_value- (float) p-value corresponding to 'statistics'
         -result_path- (str) stores path for saving results
        *title_prefix* (default: None) represents the first part of the
                       filename storing the created plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, figsize=(10, 3.5), sharex=True,
        gridspec_kw={'height_ratios': [2, 2, 1]})
    x_axis = np.arange(0, local_variances.shape[0], 1)
    displayed_p_value = display_p_values(parameters["p_value"])
    if np.isnan(parameters["statistics"]):
        correlation_string = "r is undefined"
    else:
        correlation_string = f"r={parameters['statistics']:.2f}"
    ax1.plot(x_axis,
             mean_RR_values_for_selected_individual[
                 (parameters["time_window_range"] - 1):],
             '-', color='red')
    ax1.set_ylabel('R-R value')
    ax1.set_title(f"R-R values and rolling variances in time, group: "
                  f"{parameters['group_of_individual']}, "
                  f"ID: {parameters['ID_of_individual']}, "
                  f"{correlation_string}, {displayed_p_value}",
                  fontsize=10)
    ax2.plot(x_axis, local_variances, '-', color='blue')
    ax2.set_ylabel('Rolling variance', fontsize=9)
    for i, value in enumerate(evaluation):
        if value == 1:
            ax3.axvline(x=i, color='green', linestyle='-', linewidth=1)
        else:
            ax3.axvline(x=i, color='red', linestyle='-', linewidth=1)
    ax3.set_xlabel('Index of time window')
    ax3.yaxis.set_ticklabels([])
    ax3.set_ylabel('Prediction', fontsize=9)
    plt.tight_layout()
    os.makedirs(parameters['result_path'], exist_ok=True)
    if title_prefix is None:
        title_prefix = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    plt.savefig(f"{parameters['result_path']}{title_prefix}_"
                f"{parameters['group_of_individual']}_"
                f"{parameters['ID_of_individual']}.pdf", dpi=300)
    plt.close()


def uncertainty_modeling_for_selected_person(input_individuals,
                                             individual_data,
                                             fold,
                                             seed,
                                             selected_individual,
                                             time_window_range,
                                             save_result_path,
                                             prepare_plot=True):
    """
    Perform uncertainty modeling analysis for a single person.
    Calculates rolling variances in *time_window_range* subsequent
    time windows. Also, calculates the Pearson's correlation coefficient
    between predictions and raw R-R interval data values.

    Arguments:
    ----------
      *input_individuals*: (Numpy array / pickle file) contains 
      *individual_data*: (Numpy .npz file) 
    """
    # Extract information about particular person
    group_of_individual = selected_individual[:selected_individual.find('_')]
    ID_of_individual = int(selected_individual[(selected_individual.find('_') + 1):])
    indices_of_selected_individual = np.argwhere(
        individual_data["person_IDs"] == selected_individual)
    predictions_for_selected_individual = individual_data[
        "test_predictions"][indices_of_selected_individual].flatten()
    ground_truth_for_selected_individual = individual_data["gt"][
        indices_of_selected_individual].flatten()

    # Extract original R-R interval data for this person
    RR_values_for_selected_individual = input_individuals[
        (input_individuals["group"] == group_of_individual) &
        (input_individuals["number"] == ID_of_individual)][
            "RR_values"].values
    # Calculate mean RR values for consecutive time windows
    mean_RR_values_for_selected_individual = np.array(
        [np.mean(x) for x in RR_values_for_selected_individual])
    assert predictions_for_selected_individual.shape == \
        ground_truth_for_selected_individual.shape == \
        RR_values_for_selected_individual.shape == \
        mean_RR_values_for_selected_individual.shape

    # Prepare evaluation function
    ground_truth_value = np.unique(ground_truth_for_selected_individual)
    assert len(ground_truth_value) == 1
    ground_truth_value = ground_truth_value[0]
    # Calculate Pearson's r between evaluation function and mean R-R interval values
    evaluation = np.where(predictions_for_selected_individual ==
                          ground_truth_for_selected_individual, 1, 0)
    correlation = pearsonr(evaluation, mean_RR_values_for_selected_individual)
    statistics, p_value = correlation[0], correlation[1]

    ##############
    # Perform classification based on variance minimization
    local_variances = moving_window_variances(
        predictions_for_selected_individual, time_window_range)
    minimum_variance_prediction = designate_prediction_based_on_minimum_variance(
        predictions_for_selected_individual,
        local_variances, time_window_range)
    ##############
    # Save visualizations
    results = {
        "time_window_range": time_window_range,
        "group_of_individual": group_of_individual,
        "ID_of_individual": ID_of_individual,
        "statistics": statistics,
        "p_value": p_value,
        "result_path": save_result_path,
        "minimum_variance_prediction": minimum_variance_prediction,
        "ground_truth": ground_truth_value,
    }
    if prepare_plot:
        plot_combined_variance_RR_predictions_for_individual(
            mean_RR_values_for_selected_individual,
            local_variances,
            evaluation,
            results,
            title_prefix=f"fold_{fold}_seed_{seed}_time_range_{time_window_range}"
        )
    return results


def test_designate_prediction_based_on_minimum_variance():
    """
    Unittest of designate_prediction_based_on_minimum_variance()
    """
    # Example 1)
    predictions = np.array([1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0])
    variances = moving_window_variances(predictions, 5)
    gt_patient_class = 0
    predicted_patient_class = designate_prediction_based_on_minimum_variance(
        predictions, variances, 5
    )
    assert gt_patient_class == predicted_patient_class

    # Example 2)
    predictions = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    variances = moving_window_variances(predictions, 2)
    gt_patient_class = 0
    predicted_patient_class = designate_prediction_based_on_minimum_variance(
        predictions, variances, 2
    )
    assert gt_patient_class == predicted_patient_class


if __name__ == "__main__":
    timestep = 60  # 60 or 300
    time_window_range = 60
    prepare_plot = True
    load_result_path = './Data/'
    input_data_path = f'./Data/individual_measurements_window_{timestep}_5_folds.pkl'
    input_individuals = np.load(input_data_path, allow_pickle=True)
    save_result_path = './Results/rolling_model_evaluation/'
    # List storing all results
    all_individuals_statistics = []
    # group ID, number of person, fold, seed, time window range, correlation,
    # p-value, classifier based on multiple windows, ground truth

    folds = list(range(0, 5))
    seeds = list(range(1, 6))
    my_product = product(folds, seeds)

    for fold, seed in my_product:
        load_result_file = (
            f'summary_results_GRU_FCN_part_single_training_timestep_{timestep}_'
            f'fold_{fold}_seed_{seed}_individual_preds.npz'
        )
        individual_data = np.load(f'{load_result_path}{load_result_file}',
                                  allow_pickle=True)
        get_all_person_IDs = np.unique(individual_data["person_IDs"])
        for selected_individual in get_all_person_IDs:
            single_result = uncertainty_modeling_for_selected_person(
                input_individuals,
                individual_data,
                fold,
                seed,
                selected_individual,
                time_window_range,
                save_result_path,
                prepare_plot=prepare_plot
            )
            all_individuals_statistics.append(
                [
                    single_result["group_of_individual"],
                    single_result["ID_of_individual"],
                    fold,
                    seed,
                    time_window_range,
                    single_result["statistics"],
                    single_result["p_value"],
                    single_result["minimum_variance_prediction"],
                    single_result["ground_truth"]
                ]
            )
    dataframe = pd.DataFrame(
        all_individuals_statistics,
        columns=[
            'group', 'ID', 'fold', 'seed', 'time_window_range',
            'correlation', 'p_value', 'minimum_variance_prediction',
            'ground_truth']
    )
    dataframe.to_csv(
        f"{save_result_path}summary_results_variance_correlation.csv",
        sep=';'
    )

import os


def set_hyperparameters(model, timestep_length, part="single_training"):
    """
    Set hyperparameters for grid search optimization or final evaluation
    using the best set of hyperparameters

    Arguments:
    ----------
      *model*: (str) a neural network model for evaluation ('transformer'
               or 'GRU_FCN')
      *timestep_length*: (int) the length of single timestep; (60 or 300)
      *part*: (optional int/str): select the set(s) of hyperparameters
              for testing; by default: 'single_training'

    Returns a dictionary with hyperparameters.
    """
    assert model in ["transformer", "GRU_FCN"]
    assert timestep_length in [60, 300]
    # Common part of hyperparameters
    hyperparameters = {
        "saving_folder": (
            f"./Results/recalc_final_experiment_{model}_part_{part}_"
            f"timestep_{timestep_length}"
        ),
        "saving_filename": (
            f"summary_results_{model}_part_{part}_"
            f"timestep_{timestep_length}"
        ),
        "validation_size_for_final_training": 0.1,
        "number_of_epochs": 30,
        "data_standardization": True,
        "timestep_length": timestep_length,
        "number_of_features": 1,
        "number_of_classes": 2,
        "early_stopping_patience": 10,
        "learning_rate_patience": 5,
        "model": model,
        "seeds": [3, 4],
        "batch_sizes": [64, 128],
        "learning_rates": [0.001, 0.0001],
    }

    if model == "transformer":
        hyperparameters["transformer_dimensions"] = [128, 512]
        hyperparameters["transformer_numbers_of_heads"] = [8, 16]
        hyperparameters["transformer_feedforward_dimensions"] = [256, 512]
        hyperparameters["transformer_numbers_enc_layers"] = [3]
        hyperparameters["transformer_dropouts"] = [0.1, 0.3]
        hyperparameters["RNN_hidden_sizes"] = None
        hyperparameters["RNN_numbers_of_layers"] = None
        hyperparameters["RNN_dropouts"] = None
        hyperparameters["RNN_bidirectional_opts"] = None
    elif model == "GRU_FCN":
        hyperparameters["transformer_dimensions"] = None
        hyperparameters["transformer_numbers_of_heads"] = None
        hyperparameters["transformer_feedforward_dimensions"] = None
        hyperparameters["transformer_numbers_enc_layers"] = None
        hyperparameters["transformer_dropouts"] = None
        hyperparameters["RNN_hidden_sizes"] = [100, 200]
        hyperparameters["RNN_numbers_of_layers"] = [1, 2]
        hyperparameters["RNN_dropouts"] = [0.4, 0.8]
        hyperparameters["RNN_bidirectional_opts"] = [True, False]

    if part == "grid":
        pass
    ##############################
    # Best sets of hyperparameters
    elif part in ["single_training", "evaluation"]:
        if model == "GRU_FCN" and timestep_length == 60:
            hyperparameters["seeds"] = [1, 2, 3, 4, 5]
            hyperparameters["learning_rates"] = [0.0001]
            hyperparameters["batch_sizes"] = [128]
            hyperparameters["RNN_hidden_sizes"] = [100]
            hyperparameters["RNN_numbers_of_layers"] = [1]
            hyperparameters["RNN_dropouts"] = [0.8]
            hyperparameters["RNN_bidirectional_opts"] = [False]
            hyperparameters["transformer_dimensions"] = None
            hyperparameters["transformer_numbers_of_heads"] = None
            hyperparameters["transformer_feedforward_dimensions"] = None
            hyperparameters["transformer_numbers_enc_layers"] = None
            hyperparameters["transformer_dropouts"] = None
        elif model == "GRU_FCN" and timestep_length == 300:
            hyperparameters["seeds"] = [1, 2, 3, 4, 5]
            hyperparameters["learning_rates"] = [0.001]
            hyperparameters["batch_sizes"] = [64]
            hyperparameters["RNN_hidden_sizes"] = [200]
            hyperparameters["RNN_numbers_of_layers"] = [1]
            hyperparameters["RNN_dropouts"] = [0.8]
            hyperparameters["RNN_bidirectional_opts"] = [False]
            hyperparameters["transformer_dimensions"] = None
            hyperparameters["transformer_numbers_of_heads"] = None
            hyperparameters["transformer_feedforward_dimensions"] = None
            hyperparameters["transformer_numbers_enc_layers"] = None
            hyperparameters["transformer_dropouts"] = None
        elif model == "transformer":
            # For both timestep lengths
            hyperparameters["seeds"] = [1, 2, 3, 4, 5]
            hyperparameters["learning_rates"] = [0.001]
            hyperparameters["batch_sizes"] = [64]
            hyperparameters["RNN_hidden_sizes"] = None
            hyperparameters["RNN_numbers_of_layers"] = None
            hyperparameters["RNN_dropouts"] = None
            hyperparameters["RNN_bidirectional_opts"] = None
            hyperparameters["transformer_dimensions"] = [512]
            hyperparameters["transformer_numbers_of_heads"] = [16]
            hyperparameters["transformer_feedforward_dimensions"] = [512]
            hyperparameters["transformer_numbers_enc_layers"] = [3]
            hyperparameters["transformer_dropouts"] = [0.1]
    os.makedirs(hyperparameters["saving_folder"], exist_ok=True)
    return hyperparameters


if __name__ == "__main__":
    pass

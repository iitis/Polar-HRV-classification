## Polar R-R Interval Data Classification Library (RR-DIAGNOSE) v 1.0
Library for the classification of R-R interval data based on measurements from Polar H10 wearable devices with two deep learning models.

HRV and accelerometer data analysis based on measurements from Polar H10 wearable devices.

It is a source code related to the paper:

> Machine learning approach for the classification of psychotic disorder patients using RR intervals collected with a Polar H10 sensor

Authors:
- Kamil Książek (ITAI PAS, ORCID ID: [0000-0002-0201-6220](https://orcid.org/0000-0002-0201-6220)),
- Wilhelm Masarczyk (FMS MUS, ORCID ID: [0000-0001-9516-0709](https://orcid.org/0000-0001-9516-0709)),
- Przemysław Głomb (ITAI PAS, ORCID ID: [0000-0002-0215-4674](https://orcid.org/0000-0002-0215-4674)),
- Michał Romaszewski (ITAI PAS, ORCID ID: [0000-0002-8227-929X](https://orcid.org/0000-0002-8227-929X)),
- Krisztián Buza (BBU, SHU, ORCID ID: [0000-0002-7111-6452](https://orcid.org/0000-0002-7111-6452)),
- Przemysław Sekuła (ITAI PAS, ORCID ID: [0000-0002-4599-1077](https://orcid.org/0000-0002-4599-1077)),
- Michał Cholewa (ITAI PAS, ORCID ID: [0000-0001-6549-1590](https://orcid.org/0000-0001-6549-1590)),
- Piotr Gorczyca (FMS UMS, ORCID ID: [0000-0002-9419-7988](https://orcid.org/0000-0002-9419-7988)),
- Magdalena Piegza (FMS UMS, ORCID ID: [0000-0002-8009-7118](https://orcid.org/0000-0002-8009-7118)).

*ITAI PAS* - Institute of Theoretical and Applied Informatics, Polish Academy of Sciences, Gliwice, Poland;  
*FMS UMS* - Faculty of Medical Sciences in Zabrze, Medical University of Silesia, Tarnowskie Góry, Poland;  
*PDMH* - Psychiatric Department of the Multidisciplinary Hospital, Tarnowskie Góry, Poland;  
*BBU* - Budapest Business University, Hungary;  
*SHU* - Department of Mathematics-Informatics, Sapientia Hungarian University of Transylvania, Târgu Mureș, Romania.


## LICENSE:
Copyright 2023-2024
Institute of Theoretical and Applied Informatics,
Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl

The main author of the code:
- Kamil Książek (ITAI PAS, ORCID ID: [0000-0002-0201-6220](https://orcid.org/0000-0002-0201-6220)).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.


## FUNCTIONALITY:
- Performs data classification using a gated-recurrent-unit-fully-convolutional-network hybrid model ([GRU-FCN](https://arxiv.org/pdf/1812.07683.pdf)) and a transformer model based on [tsai](https://timeseriesai.github.io/tsai/) implementation.


## RULES AND USAGE:
- Main calculations with 5-fold cross-validation experiments are performed in the `basic_classification.py` file. One can select a neural network model by setting `model` to `GRU_FCN` or `transformer`. Also, one of the three options are available for working mode: `single_training` for the training of models with a single set of hyperparameters, `grid` for performing grid search experiments, i.e. training of models with multiple sets of hyperparameters, `evaluation` for the evaluation of trained and saved network models.
- Leave-one-out cross-validation experiments are performed in the `extra_classification.py` file.
- Hyperparameters for different training modes are stored in the `hyperparameters.py` file.
- It is necessary to select a threshold to decide how many windows need to be classified as a disease to consider that a given person belongs to the treatment group. Currently, this decision is made based on the validation set. If multiple thresholds achieve the same result, a median value is selected. Also, if 50 is one of those values, it is automatically selected (majority voting).
- Preparation of 5 disjoint folds is performed in the `experiment_data_scheme.py` file.
- Plots of distributions of different folds may be prepared with the `plots.py` file.


## DATASET:
Download the dataset from the [following data repository in Zenodo](https://zenodo.org/records/8171266).
The recommended path for data samples is `data` folder.


**WARNING! Please ensure the proper paths to the dataset are set in repository files.**


## FILES:
- Files inside the `data_analysis` subfolder, come from the [Polar-HRV-data-analysis](https://github.com/iitis/Polar-HRV-data-analysis/) repository.
- `basic_classification.py`: prepares 5-fold cross validation classification experiment with hyperparameters from `hyperparameters.py` file and folds created by functions from the `experiment_data_scheme.py` file. The experiment setup may be easily modified.
- `evaluation.py`: contains functions for score calculation, including calculating overall accuracy, selecting the best hyperparameters set based on grid search results, calculating disease threshold (how many time windows of a given person need to be assigned as a disease to assume that a given person belongs to the treatment group), and assessing individual persons based on their multiple windows and a selected threshold value.
- `experiment_data_scheme.py`: contains functions for the creation of disjoint cross-validation folds in a stratified manner.
- `extra_classification.py`: prepares leave-one-out cross-validation experiments. Because the dataset contains 60 people, 60 models have to be trained. Each time, 59 people belong to the training/validation set while the remaining person is in the test set.
- `hyperparameters.py`: contains model hyperparameters, both `transformer` and `GRU_FCN` for two lengths of the time intervals: 60 and 300; also, one of the four modes may be selected: `single_training`, `evaluation`, `grid` and `test` (the last option is additional and is not implemented in `basic_classification.py` file).
- `loading.py`: contains functions for data loading, preparing training/validation/test sets, performing data standardization, etc.
- `plots.py`: contains functions for plots with data distributions.

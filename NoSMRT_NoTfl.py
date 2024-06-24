from src.evaluation_NoSMRTNoTfl import evaluate_model_NoSMRTNoTfl
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from src import preprocessing, training_NoSMRTNoTfl
from utils.stratification import stratify_y


is_smoke_test = False
is_smrt = False


number_of_folds = 5
number_of_trials = 25
param_search_folds = 5


if __name__ == "__main__":
    # Load data
    tsv_file = "./resources/fgp_no_SMRT.tsv"

    # Read the TSV file into a pandas DataFrame
    df = pd.read_csv(tsv_file, delimiter='\t')

    first_column = df.iloc[:, 0]
    first_column_modified = first_column.str.split('_').str[0].str[:4]

    # Convert the modified first column to integers
    first_column_int = first_column_modified.astype(int)

    # Find the highest number
    highest_number = first_column_int.max()

    # DELETE DUPLICATES
    # delete the duplicates and turn the variable to a list
    first_column_noDupl = list(set(first_column_int))

    missing_numbers = []
    # Iterate through the sorted numbers
    for i in range(1, len(first_column_noDupl)):
        # If the difference between consecutive numbers is not 1, there's a missing number
        if first_column_noDupl[i] - first_column_noDupl[i - 1] != 1:
            for j in range(first_column_noDupl[i - 1] + 1, first_column_noDupl[i]):
                missing_numbers.append(j)

    experimentsWithErrors = [8, 13, 16, 65, 66, 81, 123, 130, 137, 145, 146, 326, 342, 358, 359, 362]

    # for experiment in range(1, highest_number + 1):
    for experiment in range(1, highest_number+1):
        # if the number of experiment is either a missing values or gives errors continue with the next number
        if experiment in missing_numbers or experiment in experimentsWithErrors:
            continue
        # turn int to str and add the required zeros in the beginning to get 4 digits in total
        print(f"Experiment {experiment}")
        expNumber = str(experiment).zfill(4)

        # Transformation
        # keep the rows where the first column starts with '0001' (first experiment)
        filtered_experiment = df[df.iloc[:, 0].str.startswith(expNumber)]

    # Transformation
    # keep the rows where the first column starts with '0001' (first experiment)
        # filtered_experiment = df[df.iloc[:, 0].str.startswith('0001')]
        # Drop the inchi.std and id, instead of df we can write filtered_experiment
        filtered_experiment = filtered_experiment.drop(df.columns[[0, 2]], axis=1)
        # turn rt from minutes to seconds
        filtered_experiment.iloc[:, 0] *= 60
        # get the rt, in a column, MAYBE I NEED TO FLATTEN THE VALUES .values.flatten()
        y = filtered_experiment.iloc[:, 0]
        # get the features
        X = filtered_experiment.iloc[:, 1:]
        # merge the columns into one. Each row is a different feature vector
        # X = X.apply(lambda row: row.values, axis=1)
        X=X.to_numpy()
        # create the fingerprints_columns
        fingerprints_columns = np.arange( X.shape[1], dtype='int')
        # in our case we dont have descriptors so an empty list should probably work
        descriptors_columns = []
        # Necessary preformatting step
        y = np.array(y).astype('float32').flatten()
        X = X.astype('float32')

        # Do K number of folds for cross validation and save the splits into a variable called splits
        splitting_function = KFold(n_splits=number_of_folds, shuffle=True, random_state=42)
        # Generate the splits dynamically and train with all the splits

        # Create results directory for transfer learning if it doesn't exist
        if not os.path.exists(f'./results_NoSMRTNoTfl'):
            os.makedirs(f'./results_NoSMRTNoTfl/')

        for fold, (train_indexes, test_indexes) in enumerate(splitting_function.split(X, stratify_y(y))):
            # Use the indexes to actually split the dataset in training and test set.
            train_split_X = X[train_indexes]
            train_split_y = y[train_indexes]
            test_split_X = X[test_indexes]
            test_split_y = y[test_indexes]

            features_list = ["fingerprints"] #if is_smoke_test else ["fingerprints", "descriptors", "all"]
            for features in features_list:
                # Preprocess X
                preprocessed_train_split_X, preprocessed_test_split_X, preproc = preprocessing.preprocess_X(
                     descriptors_columns=descriptors_columns,
                     fingerprints_columns=fingerprints_columns,
                     train_X=train_split_X,
                     train_y=train_split_y,
                     test_X=test_split_X,
                     test_y=test_split_y,
                     features=features,
                     is_smrt=is_smrt
                )

                preprocessed_train_split_y, preprocessed_test_split_y, preproc_y = preprocessing.preprocess_y(
                    train_y=train_split_y, test_y=test_split_y
                )

                trained_NoSMRTNoTfl = training_NoSMRTNoTfl.optimize_and_train_dnn_NoSMRTNoTfl(preprocessed_train_split_X,
                                                                                              preprocessed_train_split_y,
                                                                                              param_search_folds,
                                                                                              number_of_trials, fold,
                                                                                              features, experiment)

                print("Evaluation of the model & saving of the results")
                evaluate_model_NoSMRTNoTfl(trained_NoSMRTNoTfl, preprocessed_test_split_X, preprocessed_test_split_y,
                                           preproc_y, fold, features, experiment, X)



import os

from sklearn.model_selection import KFold

from src import preprocessing, training_TfL_model
from src.evaluation_tf import evaluate_model_tf
from utils.stratification import stratify_y
from tensorflow import keras
import optuna
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json


is_smoke_test = True
is_smrt = False

if is_smoke_test:
    print("Running smoke test...")
    number_of_folds = 5
    number_of_trials = 2
    param_search_folds = 1
else:
    number_of_folds = 5
    number_of_trials = 15
    param_search_folds = 5


if __name__ == "__main__":

    # Load data
    tsv_file = "./resources/fgp_no_SMRT.tsv"

    # Read the TSV file into a pandas DataFrame
    df = pd.read_csv(tsv_file, delimiter='\t')
    # get first column, split it to keep the first 4 numbers that inticate the number of experiment
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

    experimentsWithErrors=[8,13,16,65,66,81, 123, 130,137, 145, 146, 229, 233, 326, 342, 358, 359,362]



    # for experiment in range(1, highest_number + 1):
    for experiment in range(315, highest_number + 1):
        # if the number of experiment is either a missing values or gives errors continue with the next number
        if experiment in missing_numbers or experiment in experimentsWithErrors:
            continue
        # turn int to str and add the required zeros in the beginning to get 4 digits in total
        print(f"Experiment {experiment}")
        expNumber = str(experiment).zfill(4)

        # Transformation
        # keep the rows where the first column starts with '0001' (first experiment)
        filtered_experiment = df[df.iloc[:, 0].str.startswith(expNumber)]

        # Drop the inchi.std and id, instead of df we can write filtered_experiment
        filtered_experiment = filtered_experiment.drop(df.columns[[0, 2]], axis=1)

        # turn rt from minutes to seconds
        filtered_experiment.iloc[:, 0] *= 60

        #get the rt, in a coloumn, MAYBE I NEED TO FLATTEN THE VALUES .values.flatten()
        y = filtered_experiment.iloc[:, 0]
        #get the features
        X = filtered_experiment.iloc[:, 1:]
        X=X.to_numpy()

        #create the fingerprints_columns
        fingerprints_columns = np.arange( X.shape[1], dtype='int')
        #in our case we dont have descriptors so an empty list should probably work
        descriptors_columns=[]

        # Necessary preformatting step
        y = np.array(y).astype('float32').flatten()
        X = X.astype('float32')

        # Do K number of folds for cross validation and save the splits into a variable called splits
        splitting_function = KFold(n_splits=number_of_folds, shuffle=True, random_state=42)
        # Generate the splits dynamically and train with all the splits

        # Create results directory for transfer learning if it doesn't exist
        if not os.path.exists(f'./results_tf'):
            os.makedirs(f'./results_tf')

        for fold, (train_indexes, test_indexes) in enumerate(splitting_function.split(X, stratify_y(y))):
            # Use the indexes to actually split the dataset in training and test set.
            train_split_X = X[train_indexes]
            train_split_y = y[train_indexes]
            test_split_X = X[test_indexes]
            test_split_y = y[test_indexes]

            features_list = ["fingerprints"]
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


                number_of_hidden_layers1 = 13

                config_path = f"./results/dnn-3-fingerprints/config.json"
                weights_path = f"./results/dnn-3-fingerprints/model.weights.h5"

                # Step 1: Load the JSON configuration from config.json
                with open(config_path, 'r') as json_file:
                   model_config = json_file.read()

                #Step 2: Reconstruct the model architecture from the JSON configuration
                model_new = model_from_json(model_config)


                # Step 3: Load the weights from model.weights.h5
                model_new.load_weights(weights_path)


                # freeze the weights of the input and deep intermediate layers and train the rest
                for layer in model_new.layers[:number_of_hidden_layers1+1]:
                    layer.trainable = False
                # model_new.summary()
                print('TRAINING WITH FROZEN LAYERS')
                T=0
                trained_NoSMRTNoTfl = training_TfL_model.optimize_and_train_dnn_TfL(preprocessed_train_split_X,
                                                                                    preprocessed_train_split_y,
                                                                                    number_of_trials,
                                                                                    fold, features, experiment,
                                                                                    model_new, T)
                print("Evaluation of the model & saving of the results")
                evaluate_model_tf(trained_NoSMRTNoTfl, preprocessed_test_split_X, preprocessed_test_split_y, preproc_y, fold,
                                  features, experiment, X)

                # unfreeze the already frozen layers and train again
                # we have to lower the learning rate!!!!
                for layer in trained_NoSMRTNoTfl.layers[:number_of_hidden_layers1+1]:
                    layer.trainable = True
                #  trained_NoSMRTNoTfl.summary()
                T=1
                print('TRAINING AFTER UNFREEZING')
                trained_NoSMRTNoTfl2 = training_TfL_model.optimize_and_train_dnn_TfL(preprocessed_train_split_X,
                                                                                    preprocessed_train_split_y,
                                                                                    number_of_trials,
                                                                                    fold, features, experiment,
                                                                                    trained_NoSMRTNoTfl,T)



                print("Evaluation of the model & saving of the results")
                evaluate_model_tf(trained_NoSMRTNoTfl2, preprocessed_test_split_X, preprocessed_test_split_y,
                                  preproc_y, fold, features, experiment, X)



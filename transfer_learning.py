import os

from sklearn.model_selection import StratifiedKFold

from src import preprocessing
from utils.data_loading import get_my_data
from src.evaluation import evaluate_model_tf
from utils.stratification import stratify_y
from tensorflow import keras
import optuna


is_smoke_test = True
is_smrt = True

if is_smoke_test:
    print("Running smoke test...")
    number_of_folds = 2
    number_of_trials = 2
    param_search_folds = 2
else:
    number_of_folds = 5
    number_of_trials = 15
    param_search_folds = 5


if __name__ == "__main__":
    # Load data
    print("Loading data")
    common_columns = ['pid', 'rt'] if is_smrt else ['unique_id', 'correct_ccs_avg']
    X, y, descriptors_columns, fingerprints_columns = get_my_data(common_columns=common_columns,
                                                                  is_smoke_test=is_smoke_test)

    # Create results directory if it doesn't exist
    if not os.path.exists('./results'):
        os.makedirs('./results')

    # Do K number of folds for cross validation and save the splits into a variable called splits
    splitting_function = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=42)
    # Generate the splits dynamically and train with all the splits
    for fold, (train_indexes, test_indexes) in enumerate(splitting_function.split(X, stratify_y(y))):
        # Use the indexes to actually split the dataset in training and test set.
        train_split_X = X[train_indexes]
        train_split_y = y[train_indexes]
        test_split_X = X[test_indexes]
        test_split_y = y[test_indexes]

        features_list = ["fingerprints"] if is_smoke_test else ["fingerprints", "descriptors", "all"]
        for features in features_list:
            # Preprocess X
            preprocessed_train_split_X, preprocessed_test_split_X, preproc = preprocessing.preprocess_X(
                 descriptors_columns=descriptors_columns,
                 fingerprints_columns=fingerprints_columns,
                 train_X=train_split_X,
                 train_y=train_split_y,
                 test_X=test_split_X,
                 test_y=test_split_y,
                 features=features
            )

            preprocessed_train_split_y, preprocessed_test_split_y, preproc_y = preprocessing.preprocess_y(
                train_y=train_split_y, test_y=test_split_y
            )

            study = optuna.create_study(study_name=f"foundation_cross_validation-fold-{fold}-{features}",
                                        direction='minimize',
                                        storage="sqlite:///./results/cv.db",
                                        load_if_exists=True,
                                        pruner=optuna.pruners.MedianPruner()
                                        )
            best_params = study.best_params
            number_of_hidden_layers1 = best_params["number_of_hidden_layers1"]

            # load and clone the old trained dnn
            trained_dnn=keras.models.load_model(f"./results/dnn-{fold}-{features}.keras")
            # trained_dnn = keras.models.load_model('trained_dnn' + str(fold) + '.h5')
            model_clone = keras.models.clone_model(trained_dnn)
            # copy the weights(since clone_model() does not clone the weights)
            model_clone.set_weights(model_clone.get_weights())
            model_A = model_clone
            #keras.models.Sequential(model_clone.layers[:])  # to get all the layers, not sure about this

            # freeze the weights of the input and deep intermediate layers and train the rest
            for layer in model_A.layers[:number_of_hidden_layers1]:
                layer.trainable = False
            # compile and fit

            model_A.compile(optimizer=keras.optimizers.Adam(learning_rate=10 ** (-2)),
                                    loss=keras.losses.MeanAbsoluteError(),
                                    metrics=[keras.metrics.MeanSquaredError(),
                                             keras.metrics.MeanAbsolutePercentageError()],
                                    )
            model_A.fit(
                x=preprocessed_train_split_X,
                y=preprocessed_train_split_y,
                batch_size=16,
                epochs=40,
                verbose=0
            )

            # unfreeze the already frozen layers and train again
            # no reason for that since we dont have many data
            # we have to lower the learning rate!!!!
            # for layer in model_A.layers[:number_of_hidden_layers1]:
            #    layer.trainable = True
            # compile and fit
            # model_A.compile(optimizer=keras.optimizers.Adam(learning_rate=10 ** (-5)),
            #               loss=keras.losses.MeanAbsoluteError(),
            #               metrics=[keras.metrics.MeanSquaredError(),
            #                        keras.metrics.MeanAbsolutePercentageError()],
            #               )
            # model_A.fit(
            #   x=preprocessed_train_split_X,
            #   y=preprocessed_train_split_y,
            #   batch_size=16,
            #   epochs=30,
            #   verbose=0)

            print("Saving dnn used for this fold")
            model_A.save(f"./results_tf/dnn-{fold}-{features}.keras")

            print("Evaluation of the model & saving of the results")
            evaluate_model_tf(model_A, preprocessed_test_split_X, preprocessed_test_split_y, preproc_y, fold,
                           features)

# we have to know the best parameters of the old trained dnn, we load the already existing study. since we dont know
# the name of the study we write
#study = optuna.load_study(study_name=None, storage="sqlite:///./results/cv.db", load_if_exists=True)
# this will load the latest study from the storage


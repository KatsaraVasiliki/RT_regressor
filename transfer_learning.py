import os

from sklearn.model_selection import StratifiedKFold

from src import preprocessing, training
from utils.data_loading import get_my_data
from src.evaluation import evaluate_model
from utils.stratification import stratify_y
from tensorflow import keras
import optuna


is_smoke_test = True
is_smrt = False

if is_smoke_test:
    print("Running smoke test...")
    number_of_folds = 2
    number_of_trials = 2
    param_search_folds = 2
else:
    number_of_folds = 5
    number_of_trials = 15
    param_search_folds = 5


if __name__ == "__transfer_learning__":
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


#load and clone the old trained dnn
trained_dnn = keras.models.load_model("trained_dnn.h5")
model_clone = keras.models.clone_model(trained_dnn)
model_clone.set_weights(model_clone.get_weights())#copy its weights(since clone_model() does not clone the weights)

model_A = keras.models.Sequential(model_clone.layers[:])#not sure about this

# we have to know the best parameters of the old trained dnn, we load the already existing study
study = optuna.create_study(study_name=f"foundation_cross_validation-fold-{fold}-{features}",
                                direction='minimize',
                                storage="sqlite:///./results/cv.db",
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner()
                                )

best_params = study.best_params

neurons_per_layer1=best_params["neurons_per_layer1"]

#freeze the weights of the input and deep intermidiate layers and train the rest
for layer in model_A.layers[:neurons_per_layer1]:
    layer.trainable = False

model_A_trained,best_params=training.optimize_and_train_dnn(preprocessed_train_split_X, preprocessed_train_split_y,
                                                          param_search_folds, number_of_trials, fold, features)

#unfreeze the already frozen layers and train again
# we have to lower the learning rate
for layer in model_A.layers[:neurons_per_layer1]:
    layer.trainable = True
model_A_trained,best_params=training.optimize_and_train_dnn(preprocessed_train_split_X, preprocessed_train_split_y,
                                                          param_search_folds, number_of_trials, fold, features)

print("Saving dnn used for this fold")
model_A_trained.save(f"./results/dnn-{fold}-{features}.keras")

print("Evaluation of the model & saving of the results")
evaluate_model(model_A_trained, preprocessed_test_split_X, preprocessed_test_split_y, preproc_y, fold, features)


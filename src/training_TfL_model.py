from optuna.trial import TrialState
from sklearn.model_selection import RepeatedKFold
import tensorflow.keras as keras
import numpy as np
import optuna

def create_objective_Tfl(X, y, cv, model_new, T):
    def objective_Tfl(trial):
        params = suggest_params_Tfl(trial, T)
        estimator = model_new
        for layer in estimator.layers[:13 + 4]:
            layer.trainable = False
        #estimator = create_dnn(X.shape[1], params)
        cross_val_scores = []
        for step, (train_index, test_index) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            estimator = fit_dnn_Tfl(estimator, X_train, y_train, params)
            test_metrics = estimator.evaluate(
                X_test, y_test, return_dict=True, verbose=0
            )
            # loss is MAE Score, use it as optuna metric
            score = test_metrics["loss"]
            cross_val_scores.append(score)
            intermediate_value = np.mean(cross_val_scores)
            trial.report(intermediate_value, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(cross_val_scores)
    return objective_Tfl


def optimize_and_train_dnn_TfL(preprocessed_train_split_X, preprocessed_train_split_y, param_search_folds, number_of_trials,
                           fold, features, experiment, model_new, T):
    cv = RepeatedKFold(n_splits=param_search_folds, n_repeats=1, random_state=42)
    n_trials = number_of_trials
    keep_going = False

    study = optuna.create_study(study_name=f"foundation_cross_validation-fold-{fold}-{features}-Tfl-exp{experiment}-T{T}",
                                direction='minimize',
                                storage="sqlite:///./results_tf/cv.db",
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner()
                                )

    objective = create_objective_Tfl(preprocessed_train_split_X, preprocessed_train_split_y, cv, model_new,T)
    trials = [trial for trial in study.get_trials() if trial.state in [TrialState.COMPLETE, TrialState.PRUNED]]
    if not keep_going:
        n_trials = n_trials - len(trials)
    if n_trials > 0:
        print(f"Starting {n_trials} trials")
        study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    estimator=model_new
    if T==0:
        for layer in estimator.layers[:14]:
            layer.trainable = False

        estimator = fit_dnn_Tfl(estimator,
                            preprocessed_train_split_X,
                            preprocessed_train_split_y,
                            best_params)
    else:
        for layer in estimator.layers[:14]:
            layer.trainable = True

        estimator = fit_dnn_Tfl(estimator,
                                preprocessed_train_split_X,
                                preprocessed_train_split_y,
                                best_params)

    return estimator



def suggest_params_Tfl(trial, T):
    if T==0: #trainable=false
        params = {
            'epochs': trial.suggest_int('epochs', 50, 100),
            'lr': trial.suggest_float('lr', 10**(-6), 10**(-2), log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32])
        }
    if T==1:#trainable=true
        params = {
            'epochs': trial.suggest_int('epochs', 10, 20),
            'lr': trial.suggest_float('lr', 10 ** (-6), 10 ** (-3), log=True),
            'batch_size': trial.suggest_categorical('batch_size', [ 16, 32])
        }

    return params


def fit_dnn_Tfl(dnn, X, y, optuna_params):
    dnn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=optuna_params["lr"]),
        loss=keras.losses.MeanAbsoluteError(),
        metrics=[
            keras.metrics.MeanSquaredError(),
            keras.metrics.MeanAbsolutePercentageError()
        ],
    )
    dnn.fit(
        x=X,
        y=y,
        batch_size=optuna_params["batch_size"],
        epochs=optuna_params["epochs"],
        verbose=1
    )
    return dnn



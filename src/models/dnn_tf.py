import tensorflow.keras as keras
from keras.layers import Dense, Dropout
from keras.models import Sequential


def create_dnn(n_features, optuna_params):
    neurons_per_layer1 = optuna_params["neurons_per_layer1"]
    number_of_hidden_layers1 = optuna_params["number_of_hidden_layers1"]
    activation1 = optuna_params["activation1"]
    dropout_rate1 = optuna_params["dropout_between_layers1"]

    neurons_per_layer2 = optuna_params["neurons_per_layer2"]
    number_of_hidden_layers2 = optuna_params["number_of_hidden_layers2"]
    dropout_rate2 = optuna_params["dropout_between_layers2"]

    layers = []
    # Input layer
    layers.append(Dense(neurons_per_layer1, input_dim=n_features))
    # Intermediate hidden layers
    for _ in range(1, number_of_hidden_layers1+1):
        layers.append(Dense(neurons_per_layer1, activation=activation1))
        layers.append(Dropout(dropout_rate1))
    # Intermediate hidden layers
    for _ in range(1, number_of_hidden_layers2+1):
        layers.append(Dense(neurons_per_layer2, activation=activation1))
        layers.append(Dropout(dropout_rate2))
    # Output layer
    layers.append(Dense(1))

    return Sequential(layers)






def suggest_params(trial):
    params = {
        'lr': trial.suggest_float('lr', 5*10 ** (-6), 10 ** (-4), log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16]),

        'number_of_hidden_layers1': trial.suggest_categorical('number_of_hidden_layers1', [2, 4, 6, 8]),
        'dropout_between_layers1': trial.suggest_float('dropout_between_layers1', 0, 0.5),
        'neurons_per_layer1': trial.suggest_categorical('neurons_per_layer1', [2048, 4096]),
        'epochs': trial.suggest_int('epochs', 10, 100),
        'activation1': trial.suggest_categorical('activation1', ['relu']),


        'number_of_hidden_layers2': trial.suggest_categorical('number_of_hidden_layers2', [3, 4, 6, 8]),
        'dropout_between_layers2': trial.suggest_float('dropout_between_layers2', 0, 0.1),
        'neurons_per_layer2': trial.suggest_categorical('neurons_per_layer2', [40, 60, 80])


    }
    return params


def fit_dnn(dnn, X, y, optuna_params):
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
        verbose=0
    )
    return dnn

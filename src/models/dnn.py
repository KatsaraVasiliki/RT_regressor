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
    activation2 = optuna_params["activation2"]
    dropout_rate2 = optuna_params["dropout_between_layers2"]

    layers = []
    # Input layer
    layers.append(Dense(neurons_per_layer1, input_dim=n_features))
    # Intermediate hidden layers (deep)
    for _ in range(1, number_of_hidden_layers1):
        layers.append(Dense(neurons_per_layer1, activation=activation1))
        Dropout(dropout_rate1)
    # Intermediate hidden layers (shallow)
    for _ in range(1, number_of_hidden_layers2):
        layers.append(Dense(neurons_per_layer2, activation=activation2))
        Dropout(dropout_rate2)
    # Output layer
    layers.append(Dense(1))

    return Sequential(layers)


def suggest_params(trial):
    params = {
        'lr': trial.suggest_float('lr', 10 ** (-5), 10 ** (-2), log=True),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),

        'number_of_hidden_layers1': trial.suggest_int('number_of_hidden_layers1', 2, 7),
        'dropout_between_layers1': trial.suggest_float('dropout_between_layers1', 0, 0.5),
        'neurons_per_layer1': trial.suggest_categorical('neurons_per_layer1', [512, 1024, 2048, 4096]),
        'epochs': trial.suggest_int('epochs', 10, 100),
        'activation1': trial.suggest_categorical('activation1', ['relu', 'leaky_relu', 'gelu', 'swish']),


        'number_of_hidden_layers2': trial.suggest_int('number_of_hidden_layers2', 2, 4),
        'dropout_between_layers2': trial.suggest_float('dropout_between_layers2', 0.1, 0.5),
        'neurons_per_layer2': trial.suggest_categorical('neurons_per_layer2', [50, 100, 200, 512]),
        'activation2': trial.suggest_categorical('activation2', ['relu', 'leaky_relu', 'gelu', 'swish'])

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

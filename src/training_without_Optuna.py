import tensorflow.keras as keras
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

def optimize_and_train_dnn_functional(preprocessed_train_split_X, preprocessed_train_split_y, dnn_topology):
    estimator = create_dnn_functional(preprocessed_train_split_X.shape[1], dnn_topology)
    estimator = fit_dnn(estimator,
                        preprocessed_train_split_X,
                        preprocessed_train_split_y,
                        )
    return estimator


def create_dnn_functional(n_features,dnn_topology):
    if dnn_topology == 'functional':
        activation1 = keras.activations.relu
        input_deep = keras.layers.Input(shape=(n_features,))
        layer_previous = Dense(2500, activation=activation1)(input_deep)

        for _ in range(1,3):
            new_layer=Dense(2500, activation=activation1)(layer_previous)
            layer_previous=new_layer

        for _ in range(1, 8):
            new_layer=Dense(2500, activation=activation1)(layer_previous)
            layer_previous=new_layer

        layers_deep = layer_previous

        layer_1 = Dense(2500, activation=activation1)(input_deep)
        layer_2 = Dense(2500, activation=activation1)(layer_1)
        layers_wide = layer_2

        concat = keras.layers.concatenate([layers_deep, layers_wide])
        layer_previous = Dense(80, activation=activation1)(concat)

        for _ in range(1, 3):
            new_layer=Dense(80, activation=activation1)(layer_previous)
            layer_previous=new_layer

        layers_deep_and_wide_small = layer_previous
        output = Dense(1)(layers_deep_and_wide_small)
        model = keras.Model(inputs=[input_deep], outputs=[output])
        return model

def fit_dnn(dnn, X, y):
    stop_here_please = EarlyStopping(patience=5)
    dnn.compile(optimizer=keras.optimizers.Adam(learning_rate=9*10 ** (-6)),
            loss=keras.losses.MeanAbsoluteError(),
            metrics=[
                keras.metrics.MeanSquaredError(),
                keras.metrics.MeanAbsolutePercentageError()])
    dnn.fit(
        x=X,
        y=y,
        batch_size=16,
        epochs=60,
        verbose=1,
        validation_split=0.1,
        callbacks=[stop_here_please]
        )
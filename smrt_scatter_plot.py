import os

from sklearn.model_selection import StratifiedKFold

from src import preprocessing, training
from utils.data_loading import get_my_data
from src.evaluation import evaluate_model
from utils.stratification import stratify_y
from tensorflow.keras.models import model_from_json
from BlackBox.Preprocessors import FgpPreprocessor
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

# Parameters
is_smoke_test = False
is_smrt = True
number_of_folds = 5

if __name__ == "__main__":
    # Load data
    print("Loading data")
    common_columns = ['pid', 'rt'] if is_smrt else ['unique_id', 'correct_ccs_avg']
    X, y, descriptors_columns, fingerprints_columns = get_my_data(common_columns=common_columns,
                                                                  is_smoke_test=is_smoke_test)
    # create a mask to find where rt is greater than 300s
    mask = y >= 300
    # filter and keep only the rt>300s
    y = y[mask]
    # using the same mask keep only the instances where rt>300s
    X = X[mask]

    print(y)
    # Create results directory if it doesn't exist
    if not os.path.exists('./results'):
        os.makedirs('./results')

    # Do K number of folds for cross validation and save the splits into a variable called splits

    stratify_y(y)
    print(y)
    preproc = FgpPreprocessor(fgp_cols=fingerprints_columns)
    preproc_y = QuantileTransformer(n_quantiles=1000, output_distribution='normal')

    # Load the model
    config_path = f"./results/dnn-3-fingerprints/config.json"
    weights_path = f"./results/dnn-3-fingerprints/model.weights.h5"

    # Step 1: Load the JSON configuration from config.json
    with open(config_path, 'r') as json_file:
        model_config = json_file.read()

    # Step 2: Reconstruct the model architecture from the JSON configuration
    model_new = model_from_json(model_config)
    # Step 3: Load the weights from model.weights.h5
    model_new.load_weights(weights_path)

    X_tranformed = preproc.fit_transform(X)
    y_tranformed = preproc_y.fit_transform(y.reshape(-1, 1))
    y_tranformed.flatten()

    preproc_y_preds = model_new.predict(X_tranformed, verbose=0)

    y_preds = preproc_y.inverse_transform(preproc_y_preds.reshape(-1, 1)).flatten()

    #test_split_y = preproc_y.inverse_transform(y_tranformed.reshape(-1, 1)).flatten()
    df = pd.DataFrame({
        'predicted': y_preds,
        'true': y})

    print(df)

    correlation_matrix = df.corr()

    # Extract correlation coefficient between 'predicted' and 'true' columns
    correlation_coefficient_pd = correlation_matrix.loc['predicted', 'true']

    print(f"Pearson correlation coefficient: {correlation_coefficient_pd}")

    correlation_coefficient_sp, p_value = pearsonr(y_preds, y)

    print(f"Pearson correlation coefficient: {correlation_coefficient_sp}")
    print(f"P-value: {p_value}")

    plt.scatter(y, y_preds, alpha=0.3)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of Actual vs Predicted RT of SMRT dataset')
    plt.show()

import os
from utils.data_loading import get_my_data
from utils.stratification import stratify_y
from tensorflow.keras.models import model_from_json
from BlackBox.Preprocessors import FgpPreprocessor
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import scipy.stats as stats
import seaborn as sns

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


    # Create results directory if it doesn't exist
    if not os.path.exists('./results'):
        os.makedirs('./results')

    # Do K number of folds for cross validation and save the splits into a variable called splits


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

    # correlation_matrix = df.corr()

    # Extract correlation coefficient between 'predicted' and 'true' columns
    # correlation_coefficient_pd = correlation_matrix.loc['predicted', 'true']
   # print(f"Pearson correlation coefficient: {correlation_coefficient_pd}")

    # Calculate the correlation coefficient
    correlation_coef, _ = stats.pearsonr(y, y_preds)

    # Create a scatter plot with the correlation coefficient in the title
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_preds, alpha=0.3)
    sns.regplot(x=y, y=y_preds, scatter=False, color='red', line_kws={'linewidth': 2})
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.title(f'Scatter Plot of Actual vs Predicted Values\nCorrelation Coefficient: {correlation_coef:.2f}', fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(['Scatter Plot: Shows the relationship between actual and predicted values.'
                   ,'Regression Line: Indicates the best fit line through the data points, giving a sense of the trend.'
                   ],
                  loc='best', fontsize=12, prop={'size': 12})
    #plt.legend(["Scatter plot", "Regression line"], loc="lower right")
    plt.show()

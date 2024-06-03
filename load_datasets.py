import pandas as pd
import numpy as np


tsv_file = "./resources/fgp_no_SMRT.tsv"

# Read the TSV file into a pandas DataFrame
df = pd.read_csv(tsv_file, delimiter='\t')
# get first column, split it to keep the first 4 numbers that intigate the number of experiment
first_column = df.iloc[:, 0]
first_column_modified = first_column.str.split('_').str[0].str[:4]

# Convert the modified first column to integers
first_column_int = first_column_modified.astype(int)

# Find the highest number
highest_number = first_column_int.max()

# delete the dublicates and turn the variable to list
first_column_noDubl=list(set(first_column_int))
missing_numbers = []
# Iterate through the sorted numbers
for i in range(1, len(first_column_noDubl)):
    # If the difference between consecutive numbers is not 1, there's a missing number
    if first_column_noDubl[i] - first_column_noDubl[i - 1] != 1:
        for j in range(first_column_noDubl[i - 1] + 1, first_column_noDubl[i]):
            missing_numbers.append(j)

# Example usage:



highest_number_padded = str(highest_number).zfill(4)
for experiment in range(1, highest_number + 1):
filtered_experiment = df[df.iloc[:, 0].str.startswith(highest_number_padded)]
# Display the highest number
print("\nThe highest number in the first column is:", highest_number)



#Transformation
# keep the rows where the first column starts with '0001' (first experiment)
filtered_experiment = df[df.iloc[:, 0].str.startswith('0001')]
#Drop the inchi.std and id
filtered_experiment=filtered_experiment.drop(df.columns[[0,2]], axis=1)
#turn rt from minutes to seconds
filtered_experiment.iloc[:, 0] *= 60
# get the rt and set it as y
y=filtered_experiment.iloc[:, 0]
# y.values.flatten()
X = filtered_experiment.iloc[:, 1:]
X.to_numpy()

#X=X.apply(lambda row: row.values, axis=1)
#  fgp_cols=pd.DataFrame({'row_number': range(len(X))})
fgp_cols = np.arange( X.shape[0], dtype='int')

# Do this necessary preformatting step
X= X.astype('float32')
y = np.array(y).astype('float32').flatten()
# Print or do something with the filtered DataFrame
print(filtered_experiment)
print(y)
print(X)
second_row = X.iloc[1]
print(second_row)
print(fgp_cols)
import pickle
import bz2
import csv

def load_picklez(file_path):
    with bz2.BZ2File(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def save_to_csv(data, csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)

# Replace 'your_file.pklz' with the path to your .pklz file
file_path = "./resources/descriptors_and_fingerprints.pklz"

# Load data from the pickle file
data = load_picklez(file_path)
X, y, desc_cols, fgp_cols =data
print(X)
print(y)
print(desc_cols)
print(fgp_cols)

# Replace 'output.csv' with the desired CSV file name
csv_file = 'descAndFing1.csv'

# Save the data to a CSV file
save_to_csv(data, csv_file)

# If you want to display the content of the CSV in the console
# You can also skip this step if you just want to save it to a file
with open(csv_file, 'r') as f:
    print(f.read())
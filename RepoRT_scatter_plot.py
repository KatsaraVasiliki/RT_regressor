import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('./results_tf/evaluation_results_tf.txt')




#plot to compare if unfreezing the layers works better than keeping them frozen

df = pd.read_csv('./results_tf/evaluation_results_tf.txt')
#comparing frozen and unfrozen layers
grouped = df.groupby(['experiment_number', 'fold'])
df_cleaned = df.groupby(['experiment_number', 'fold']).tail(2)


df_frozen = df_cleaned.groupby(['experiment_number', 'fold']).nth(0).reset_index()
df_unfrozen = df_cleaned.groupby(['experiment_number', 'fold']).nth(1).reset_index()

# Calculate average errors for each experiment for frozen results
avg_frozen = df_frozen.groupby('experiment_number').agg({
    'mae': 'mean',
    'medae': 'mean',
    'mape': 'mean'
}).reset_index()

# Calculate average errors for each experiment for unfrozen results
avg_unfrozen = df_unfrozen.groupby('experiment_number').agg({
    'mae': 'mean',
    'medae': 'mean',
    'mape': 'mean'
}).reset_index()

# Rename columns to indicate frozen and unfrozen
avg_frozen.columns = ['experiment_number', 'mae_frozen', 'medae_frozen', 'mape_frozen']
avg_unfrozen.columns = ['experiment_number', 'mae_unfrozen', 'medae_unfrozen', 'mape_unfrozen']

# Merge the results into a single DataFrame
avg_errors = pd.merge(avg_frozen, avg_unfrozen, on='experiment_number')

# Calculate differences
avg_errors['MAE_diff'] = avg_errors['mae_frozen'] - avg_errors['mae_unfrozen']
avg_errors['MedAE_diff'] = avg_errors['medae_frozen'] - avg_errors['medae_unfrozen']
avg_errors['MAPE_diff'] = avg_errors['mape_frozen'] - avg_errors['mape_unfrozen']

print(avg_errors)



plt.figure(figsize=(10, 6))
plt.bar(avg_errors['experiment_number'] - 0.2, avg_errors['MAE_diff'], width=0.3, label='MAE_diff')
plt.bar(avg_errors['experiment_number'], avg_errors['MedAE_diff'], width=0.3, label='MedAE_diff')
plt.bar(avg_errors['experiment_number'] + 0.2, avg_errors['MAPE_diff'], width=0.3, label='MAPE_diff')
plt.xlabel('Experiment Number')
plt.ylabel('Difference')
plt.title('Average Differences of each metric per experiment between Frozen and Unfrozen Approaches')
plt.xticks(avg_errors['experiment_number'])
plt.legend()
plt.tight_layout()
plt.show()



#plot the size of the dataset vs the metrics
df=pd.read_csv('./results_tf/evaluation_results_tf.txt')
#find the avergare of all the folds
df_cleaned = df.groupby(['experiment_number', 'fold']).tail(2)
mask = df_cleaned.index % 2 != 1
# Use the mask to select rows and drop them
df_filtered = df_cleaned[mask]

average_values = df_filtered.groupby('experiment_number')[['mae', 'medae', 'mape','dataset size´']].mean()
# for mae
plt.scatter(average_values.iloc[:,-1], average_values.iloc[:,0], alpha=0.5)
plt.xlabel('dataset size')
plt.ylabel('mae')
plt.title('scatter Plot of the size of the dataset vs their MAE')
plt.show()
# for medae
plt.scatter(average_values.iloc[:,-1], average_values.iloc[:,1], alpha=0.5)
plt.xlabel('dataset size')
plt.ylabel('medae')
plt.title('scatter Plot of the size of the dataset vs their medae')
plt.show()
# for mape
plt.scatter(average_values.iloc[:,-1], average_values.iloc[:,2], alpha=0.5)
plt.xlabel('dataset size')
plt.ylabel('mape')
plt.title('scatter Plot of the size of the dataset vs their MAPE')
plt.show()



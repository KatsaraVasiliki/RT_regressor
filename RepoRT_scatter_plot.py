import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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



plt.figure(figsize=(12, 6))
plt.bar(avg_errors['experiment_number'] - 0.2, avg_errors['MAE_diff'], width=0.3, label='MAE_diff')
plt.bar(avg_errors['experiment_number'], avg_errors['MedAE_diff'], width=0.3, label='MedAE_diff')
plt.bar(avg_errors['experiment_number'] + 0.2, avg_errors['MAPE_diff'], width=0.3, label='MAPE_diff(%)')
plt.xlabel('Experiment Number',fontsize=16)
plt.ylabel('Difference',fontsize=16)
plt.title('Average Differences of each metric per experiment between Frozen and Unfrozen Approaches',fontsize=16)
plt.xticks(avg_errors['experiment_number'])
plt.xticks(ticks=np.arange(0, 100, 5), fontsize=14)
plt.yticks(ticks=np.arange(-350, 150, 50), fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()


tfl=1
#plot the size of the dataset vs the metrics
if tfl==1:
    df=pd.read_csv('./results_tf/evaluation_results_tf_unfrozen.txt')
else:
    df=pd.read_csv("./results_NoSMRTNoTfl/evaluation_results_NoSMRTNoTfl.txt")

#find the avergare of all the folds
average_values = df.groupby('experiment_number')[['mae', 'medae', 'mape','dataset size']].mean()
## for mae
#plt.scatter(average_values.iloc[:,-1], average_values.iloc[:,0], alpha=0.5)
#plt.xlabel('dataset size')
#plt.ylabel('mae')
#plt.title('scatter Plot of the size of the dataset vs their MAE')
#plt.show()
# for medae
#plt.scatter(average_values.iloc[:,-1], average_values.iloc[:,1], alpha=0.5)
#plt.xlabel('dataset size', fontsize=14)
#plt.ylabel('medae', fontsize=14)
#plt.title('scatter Plot of the size of the dataset vs their medae', fontsize=14)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#plt.show()
# for mape
#plt.scatter(average_values.iloc[:,-1], average_values.iloc[:,2], alpha=0.5)
#plt.xlabel('dataset size')
#plt.ylabel('mape')
#plt.title('scatter Plot of the size of the dataset vs their MAPE')
#plt.show()



#if we want then in subplot
fig, axs = plt.subplots(1, 3, figsize=(23, 6))
# Create subplots with increased height space
#fig, axs = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'hspace': 0.5})
# Scatter plot on the first subplot
axs[0].scatter(average_values.iloc[:,-1], average_values.iloc[:,0], alpha=0.5)
axs[0].set_xlabel('Dataset size',fontsize=20)
axs[0].set_ylabel('MAE',fontsize=20)
axs[0].set_title('Scatter plot of the size of each dataset vs their MAE',fontsize=20)

# Scatter plot on the second subplot
axs[1].scatter(average_values.iloc[:,-1], average_values.iloc[:,1], alpha=0.5)
axs[1].set_title('Scatter plot of the size of each dataset vs their MedAE',fontsize=20)
axs[1].set_xlabel('Dataset size',fontsize=20)
axs[1].set_ylabel('MedAE',fontsize=20)


# Scatter plot on the third subplot
axs[2].scatter(average_values.iloc[:,-1], average_values.iloc[:,2], alpha=0.5)
axs[2].set_title('Scatter plot of the size of each dataset vs their MAPE',fontsize=20)
axs[2].set_xlabel('Dataset size',fontsize=20)
axs[2].set_ylabel('MAPE(%)',fontsize=20)


#plot the size of the dataset vs the metrics
if tfl==1:
    fig.suptitle('Scatter Plots of Dataset Size vs Error Metrics for training with transfer learning', fontsize=24)
else:
    fig.suptitle('Scatter Plots of Dataset Size vs Error Metrics for training without transfer learning', fontsize=24)


# Adjust layout to prevent overlap
plt.tight_layout()
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=18)
# Display the subplots
plt.show()
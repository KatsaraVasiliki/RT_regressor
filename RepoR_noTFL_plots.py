import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df_TFL=pd.read_csv('./results_tf/evaluation_results_tf_unfrozen.txt')
df_noTFL=pd.read_csv("./results_NoSMRTNoTfl/evaluation_results_NoSMRTNoTfl.txt")

#plot to compare transfer learning with direct training

# Calculate average errors for each experiment for tfl
avg_TFL = df_TFL.groupby('experiment_number').agg({
    'mae': 'mean',
    'medae': 'mean',
    'mape': 'mean',
    'dataset size': 'first'
}).reset_index()

# Calculate average errors for each experiment for notfl
avg_noTFL= df_noTFL.groupby('experiment_number').agg({
    'mae': 'mean',
    'medae': 'mean',
    'mape': 'mean',
    'dataset size': 'first'
}).reset_index()

# Rename columns to indicate tfl and no tfl
avg_TFL.columns = ['experiment_number', 'mae_TFL', 'medae_TFL', 'mape_TFL', 'dataset size']
avg_noTFL.columns = ['experiment_number', 'mae_noTFL', 'medae_noTFL', 'mape_noTFL', 'dataset size']

# Merge the results into a single DataFrame
avg_error_dif = pd.merge(avg_TFL, avg_noTFL, on='experiment_number')

# Calculate differences
avg_error_dif['MAE_diff'] = avg_error_dif['mae_TFL'] - avg_error_dif['mae_noTFL']
avg_error_dif['MedAE_diff'] = avg_error_dif['medae_TFL'] - avg_error_dif['medae_noTFL']
avg_error_dif['MAPE_diff'] = avg_error_dif['mape_TFL'] - avg_error_dif['mape_noTFL']

print(avg_error_dif)


plt.figure(figsize=(12, 6))
plt.bar(avg_error_dif['experiment_number'] - 0.2, avg_error_dif['MAE_diff'], width=0.3, label='MAE_diff')
plt.bar(avg_error_dif['experiment_number'], avg_error_dif['MedAE_diff'], width=0.3, label='MedAE_diff')
plt.bar(avg_error_dif['experiment_number'] + 0.2, avg_error_dif['MAPE_diff'], width=0.3, label='MAPE_diff(%)')
plt.xlabel('Experiment Number',fontsize=16)
plt.ylabel('Difference',fontsize=16)
plt.title('Average Differences of each metric per experiment between training with transfer learning and without',fontsize=14)
plt.xticks(avg_error_dif['experiment_number'])
plt.xticks(ticks=np.arange(0, 101, 5), fontsize=14)
plt.yticks(ticks=np.arange(-350, 150, 50), fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

## bar plot
##FOR MAE
plt.figure(figsize=(10, 6))
plt.bar(avg_error_dif['experiment_number'], avg_error_dif['mae_TFL'], width=0.6, label='Transfer learning')
plt.bar(avg_error_dif['experiment_number'] + 0.4, avg_error_dif['mae_noTFL'], width=0.6, label='No transfer learning')
plt.xlabel('Experiment Number',fontsize=16)
plt.ylabel('MAE', fontsize=16)
plt.title('Comparison of MAE Between Transfer learning and no Transfer learning', fontsize=16)
plt.legend()
#plt.xticks(avg_error_dif['experiment_number'])
plt.xticks(ticks=np.arange(0, 101, 5), fontsize=14)
plt.yticks(ticks=np.arange(0, 900, 200), fontsize=14)
plt.grid(True)
plt.show()
##FOR MEDAE
plt.figure(figsize=(10, 6))
plt.bar(avg_error_dif['experiment_number'], avg_error_dif['medae_TFL'], width=0.6, label='Transfer learning')
plt.bar(avg_error_dif['experiment_number'] + 0.4, avg_error_dif['medae_noTFL'], width=0.6, label='No transfer learning')
plt.xlabel('Experiment Number',fontsize=16)
plt.ylabel('MdeAE', fontsize=16)
plt.title('Comparison of MedAE Between Transfer learning and no Transfer learning', fontsize=16)
plt.legend()
#plt.xticks(avg_error_dif['experiment_number'])
plt.xticks(ticks=np.arange(0, 101, 5), fontsize=14)
plt.yticks(ticks=np.arange(0, 900, 200), fontsize=14)
plt.grid(True)
plt.show()
##For mape
plt.figure(figsize=(10, 6))
plt.bar(avg_error_dif['experiment_number'], avg_error_dif['mape_TFL'], width=0.6, label='Transfer learning')
plt.bar(avg_error_dif['experiment_number'] + 0.4, avg_error_dif['mape_noTFL'], width=0.6, label='No transfer learning')
plt.xlabel('Experiment Number',fontsize=16)
plt.ylabel('MAPE(%)', fontsize=16)
plt.title('Comparison of MAPE Between Transfer learning and no Transfer learning', fontsize=16)
plt.legend()
#plt.xticks(avg_error_dif['experiment_number'])
#plt.xticks(ticks=np.arange(0, 100, 5), fontsize=14)
#plt.yticks(ticks=np.arange(0, 900, 200), fontsize=14)
plt.grid(True)
plt.show()



#box plot
# Create a figure and axis
plt.figure(figsize=(12, 6))

# Loop through each metric and create a boxplot for each
metrics = ['mae', 'medae', 'mape']
for i, metric in enumerate(metrics, start=1):
    plt.subplot(1, 3, i)
    plt.boxplot([
        avg_error_dif[f'{metric}_TFL'],
        avg_error_dif[f'{metric}_noTFL']
    ],
    labels=['TFL', 'noTFL'],
    patch_artist=True,  # Fills the box with color
    notch=True,         # Shows the notch
    showmeans=True,     # Shows the mean
    meanline=True,      # Draws a line at the mean
    sym='o',            # Symbol for outliers
    widths=0.6,
    showfliers=False)         # Width of the boxes

    plt.title(f'{metric.upper()} Comparison')
    plt.ylabel(f'{metric.upper()} Value')
    plt.grid(True)

plt.tight_layout()
plt.show()


##### Comparing the overall performance

average_metrics = avg_error_dif[['mae_TFL', 'medae_TFL', 'mape_TFL', 'mae_noTFL', 'medae_noTFL', 'mape_noTFL']].mean()
comparison_df = pd.DataFrame({
    'Metric': ['MAE', 'MedAE', 'MAPE'],
    'Average_TFL': [average_metrics['mae_TFL'], average_metrics['medae_TFL'], average_metrics['mape_TFL']],
    'Average_noTFL': [average_metrics['mae_noTFL'], average_metrics['medae_noTFL'], average_metrics['mape_noTFL']]
})
print(comparison_df)
#comparison_df.plot(kind='bar', x='Metric', y=['Average_TFL', 'Average_noTFL'], figsize=(10, 6))
#plt.title('Comparison of Average Metrics across all experiments for TFL and noTFL Models', fontsize=16)
#plt.ylabel('Average Value',fontsize=16)
#plt.xlabel('Metric', fontsize=16)
#plt.xticks(rotation=0)
#plt.legend(['TFL', 'No TFL'],fontsize=14)
#plt.xticks(rotation=0, fontsize=14)
#plt.yticks(fontsize=14)
#plt.grid(True)
#plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics = ['MAE', 'MedAE', 'MAPE']
for i, metric in enumerate(metrics):
    # Select the corresponding data for the metric
    df_metric = comparison_df[comparison_df['Metric'] == metric]
    df_metric.plot(kind='bar', x='Metric', y=['Average_TFL', 'Average_noTFL'], ax= axes[i], legend=False)
    axes[i].set_title(f'Comparison of {metric} for TFL and noTFL Models', fontsize=16)
    axes[i].set_ylabel('Average Value', fontsize=16)
    axes[i].set_xlabel(f'{metric}', fontsize=16)
    axes[i].set_xticks([])
    axes[i].tick_params(axis='x', labelsize=14)
    axes[i].tick_params(axis='y', labelsize=14)
    axes[i].grid(True)
# Add overall title to the entire figure
fig.suptitle('Comparison of Average Metrics across all experiments for TFL and noTFL Models', fontsize=20)
# Add legend to the last axis
axes[-1].legend(['TFL', 'No TFL'], fontsize=14)
plt.xticks(rotation=0, fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


##For dataset<100
#avg_TFL2 = avg_TFL[avg_TFL['dataset size'] > 100]
#avg_noTFL2 = avg_noTFL[avg_noTFL['dataset size'] > 100]
#avg_error_dif2 = pd.merge(avg_TFL2, avg_noTFL2, on='experiment_number')
#average_metrics2 = avg_error_dif2[['mae_TFL', 'medae_TFL', 'mape_TFL', 'mae_noTFL', 'medae_noTFL', 'mape_noTFL']].mean()
#comparison_df2 = pd.DataFrame({
#    'Metric': ['MAE', 'MedAE', 'MAPE'],
#    'Average_TFL': [average_metrics2['mae_TFL'], average_metrics2['medae_TFL'], average_metrics2['mape_TFL']],
#    'Average_noTFL': [average_metrics2['mae_noTFL'], average_metrics2['medae_noTFL'], average_metrics2['mape_noTFL']]})
#print(comparison_df2)


# Create a bar plot to compare average metrics
#plt.figure(figsize=(10, 6))
#average_metrics.plot(kind='bar', color=['blue', 'green', 'orange'], alpha=0.75)

#plt.title('Average Metrics for Dataset Size < 200', fontsize=16)
#plt.ylabel('Average Value', fontsize=14)
#plt.xlabel('Metric', fontsize=14)
#plt.xticks(rotation=0)
#plt.grid(True)
#plt.legend(['MAE', 'MedAE', 'MAPE'])




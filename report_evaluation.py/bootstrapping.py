import pandas as pd
import numpy as np
import os

# Set a fixed random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Load the data from the CSV file
file_path = '/home/chenzhixuan/Workspace/R2GenGPT/results/tmpclip_b_iu/instance_ce_evaluation_results.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Dynamically get all columns except 'Image_ID'
columns_to_bootstrap = [col for col in data.columns if col != 'Image_ID']

# Determine the file type based on the file path
file_name = os.path.basename(file_path)
# remove "instance_" prefix
file_name = file_name.replace("instance_", "")
if "ce" in file_name.lower():
    file_type = "ce"
elif "nlg" in file_name.lower():
    file_type = "nlg"
else:
    file_type = "results"

# Bootstrap sampling function
def bootstrap_ci(data_column, n_bootstrap=1000, ci=95):
    means = []
    n = len(data_column)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data_column, size=n, replace=True)
        means.append(np.mean(sample))
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile
    return means, np.mean(means), np.percentile(means, lower_percentile), np.percentile(means, upper_percentile)

# Compute bootstrap means and CIs for each column
bootstrap_results = {}
each_bootstrap_results = {}
for col in columns_to_bootstrap:
    means, mean, lower, upper = bootstrap_ci(data[col].dropna())
    bootstrap_results[col] = {'mean': mean, 'ci_lower': lower, 'ci_upper': upper}
    each_bootstrap_results[col] = means

# Convert the results to a DataFrame for easier visualization
bootstrap_results_df = pd.DataFrame(bootstrap_results).T
bootstrap_results_df.index.name = 'Metric'

# Save the bootstrap results to a CSV file
save_dir = os.path.dirname(file_path)
bootstrap_file_name = f"bootstrap_{file_type}_results.csv"
bootstrap_file_path = os.path.join(save_dir, bootstrap_file_name)
bootstrap_results_df.to_csv(bootstrap_file_path)

# Save each sample's results to a CSV file
each_bootstrap_file_name = f"each_bootstrap_{file_type}_results.csv"
each_bootstrap_file_path = os.path.join(save_dir, each_bootstrap_file_name)
each_bootstrap_results_df = pd.DataFrame(each_bootstrap_results)
each_bootstrap_results_df.to_csv(each_bootstrap_file_path)

print(f"Bootstrap results saved to: {bootstrap_file_path}")
print(f"Each bootstrap results saved to: {each_bootstrap_file_path}")
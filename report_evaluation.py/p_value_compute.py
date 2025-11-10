from scipy.stats import wilcoxon
import pandas as pd
import os

# Load each bootstrapped result
file1 = "/home/chenzhixuan/Workspace/R2GenGPT/results/tmpclip_l_iu/each_bootstrap_ce_results.csv"
file2 = "/home/chenzhixuan/Workspace/R2GenGPT/results/clip_iu/each_bootstrap_ce_results.csv"
tmpclip_l_results = pd.read_csv(file1)
clip_results = pd.read_csv(file2)

# Exclude the first column and 'ce_num_examples' if it exists
columns_to_compare = [col for col in tmpclip_l_results.columns if col != tmpclip_l_results.columns[0] and col != 'ce_num_examples']

# Compute p-values for each metric
p_values = {}
for col in columns_to_compare:
    data1 = tmpclip_l_results[col].dropna()
    data2 = clip_results[col].dropna()

    if len(data1) == len(data2):  # Ensure the lengths match for Wilcoxon test
        _, p_value = wilcoxon(data1, data2, alternative="greater")
        p_values[col] = p_value
    else:
        p_values[col] = None  # Mark as None if lengths don't match

# Convert the results to a DataFrame
p_values_df = pd.DataFrame.from_dict(p_values, orient='index', columns=['p-value'])
p_values_df.index.name = 'Metric'

# Save the p-values to a CSV file
output_dir = "/home/chenzhixuan/Workspace/R2GenGPT/results"
output_file = os.path.join(output_dir, "ce_p_values_results.csv")
p_values_df.to_csv(output_file)

print(f"P-values saved to: {output_file}")
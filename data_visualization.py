import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

#-------------------------------------heatmap plot---------------------------------

# Load the predictions file
file_path = 'output/predictions.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
data.head()

# Compute the correlation matrix
correlation_matrix = data.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')

# Ensure the output directory exists
output_dir = 'output/plot/'
os.makedirs(output_dir, exist_ok=True)

# Save the heatmap
heatmap_path = os.path.join(output_dir, 'heatmap.png')
plt.savefig(heatmap_path)

print(f'Heatmap saved to {heatmap_path}')



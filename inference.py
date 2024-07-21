import os
import pickle
import pandas as pd
from pathlib import Path


# Define paths
models_dir = Path('output/text_clf/models')
new_data_file = "data/data_for_preds.csv"
model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]

# initialize
new_data = pd.read_csv(new_data_file)
all_predictions = []

for model_file in model_files:
    model_path = models_dir / model_file

    # Load the model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Make predictions on the new data
    preds = model.predict(new_data)
    all_predictions.append(preds)

# Convert predictions to DataFrame
# Each column will be the predictions from one model
predictions_df = pd.DataFrame(all_predictions).T  # Transpose to make each model's predictions a column

# Add column names for each model
predictions_df.columns = [f'Model_{i+1}' for i in range(len(model_files))]

# Save predictions to CSV
predictions_csv_file = 'output/predictions.csv'
predictions_df.to_csv(predictions_csv_file, index=False)
print(f"Predictions saved to {predictions_csv_file}")

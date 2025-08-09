import os
import glob
import shutil

# Remove files matching data/*_data.csv but not data.csv
for file_path in glob.glob('data/*_data.csv'):
    if os.path.basename(file_path) != 'data.csv':
        os.remove(file_path)
        print(f"Removed: {file_path}")

# Remove trained_model/ directory if it exists
if os.path.isdir('trained_model'):
    shutil.rmtree('trained_model')
    print("Removed: trained_model/ directory")
else:
    print("No trained_model/ directory to remove")

# Remove prediction files (prediction.csv, predictions.csv, pred.csv)
prediction_files = ['prediction.csv', 'predictions.csv', 'pred.csv']
removed_predictions = []
for pred_file in prediction_files:
    if os.path.isfile(pred_file):
        os.remove(pred_file)
        removed_predictions.append(pred_file)

if removed_predictions:
    for file in removed_predictions:
        print(f"Removed: {file}")
else:
    print("No prediction files to remove")

print("Cleanup complete. All temporary files and directories have been removed.")

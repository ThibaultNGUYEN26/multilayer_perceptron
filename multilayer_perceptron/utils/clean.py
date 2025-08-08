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

print("Cleanup complete. All temporary files and directories have been removed.")

import os
import pandas as pd

# Define the header you want to add
header = [
    "generation", " map", " width", " height", " algorithm", " crossover",
    " mutation", " popsize", " n_eval", " samplingFunction", " repairFunction",
    " shiftingMethod", " seed", " objectiveValues", " paths"
]

# Define the path to the main folder
main_folder_path = '/Users/carlonue/Documents/Venvs/pathpave_pymoo/extra/sinusoidalObstacles'  # Replace with the path to main folder

# Walk through each subfolder in the main folder
for root, dirs, files in os.walk(main_folder_path):
    for file in files:
        if file == 'optLog.csv':
            # Construct the full file path
            file_path = os.path.join(root, file)
            
            # Read the CSV file without header
            df = pd.read_csv(file_path, header=None)
            
            # Add the header to the DataFrame
            df.columns = header
            
            # Save the DataFrame back to a CSV file
            df.to_csv(file_path, index=False)

print("Headers added to all optLog.csv files.")

import os
import pandas as pd # type: ignore
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

def format_objective_values(df):
    df[" objectiveValues"] = df[" objectiveValues"].str.split("array")
    
    for x in range(len(df[" objectiveValues"])):
        del df[" objectiveValues"][x][0]
        for y in range(len(df[" objectiveValues"][x])):
            df[" objectiveValues"][x][y] = df[" objectiveValues"][x][y].replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(" ", "")
            df[" objectiveValues"][x][y] = df[" objectiveValues"][x][y].rstrip(',')
            parts = df[" objectiveValues"][x][y].split(',')
            try:
                df[" objectiveValues"][x][y] = [float(part.strip()) for part in parts]
            except ValueError:
                print(f"Conversion error for value: {df[' objectiveValues'][x][y]}")
    return df

def parse_tuples(s):
    tuples = re.findall(r'\((\d+), (\d+)\)', s)
    return [(int(x), int(y)) for x, y in tuples]

def format_paths(df):
    df[' paths'] = df[' paths'].apply(parse_tuples)
    return df

def format_log_file(log_file_path):
    # Read the log.csv of the current subfolder into a DataFrame
    df = pd.read_csv(log_file_path)
    
    # Apply both formatting functions
    df1 = format_paths(df)
    df2 = format_objective_values(df1)
    return df2

def process_subfolder(subfolder_path, output_folder=None):
    csv_files = ['log.csv', 'optLog.csv']  # List of CSV files to process
    
    for csv_file in csv_files:
        file_path = os.path.join(subfolder_path, csv_file)
        if os.path.isfile(file_path):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            
            # Check if "array" is in any entry of the " objectiveValues" column
            if df[" objectiveValues"].str.contains("array").any():
                # Process and overwrite the CSV file only if "array" is found
                df = format_log_file(file_path)
                df.to_csv(file_path, index=False)
                
                # Print the file name indicating it has been processed
                print(f"Finished processing and overwriting: {file_path}")
            else:
                # Skip processing if "array" is not found
                print(f"Skipped processing: {file_path} (no 'array' in ' objectiveValues')")
    
    return None



def find_subfolders(log_folder_path):
    subfolders = []
    for root, dirs, files in os.walk(log_folder_path):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            subfolders.append(subfolder_path)
    return subfolders

def main(log_folder_path):
    subfolders = find_subfolders(log_folder_path)

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_subfolder, subfolder): subfolder for subfolder in subfolders}
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    # Print the paths of the saved CSV files
    print("Saved CSV files")

if __name__ == "__main__":
    # Set your log folder path here
    log_folder_path = "/Users/carlonue/Documents/Venvs/pathpave_pymoo/bugfree S3 shift data/sinusoidalObstacles"  # Replace with the actual path to your log folder

    main(log_folder_path)

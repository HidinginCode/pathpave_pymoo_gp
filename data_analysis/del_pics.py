import os

# Define the main folder path
main_folder_path = 'insert folder path here'

# Loop through each subfolder in the main folder
for subfolder in os.listdir(main_folder_path):
    subfolder_path = os.path.join(main_folder_path, subfolder)
    
    # Check if it's a directory
    if os.path.isdir(subfolder_path):
        # Loop through the files in the subfolder
        for file_name in os.listdir(subfolder_path):
            # If the file is a .png file, delete it
            if file_name.endswith('.png'):
                file_path = os.path.join(subfolder_path, file_name)
                os.remove(file_path)
                print(f"Deleted {file_path}")
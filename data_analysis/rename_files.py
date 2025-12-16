import os

# Define the parent folder path where the subfolders are located
parent_folder_path = '/Users/carlonue/Documents/Venvs/pathpave_pymoo/bugfree S3 shift data/sinusoidalObstacles'

# Define the part of the folder name to remove
#part_to_remove = "_OnePointCrossover_RadiusSamplingMutation_100"
#part_to_remove = "_51_51"
part_to_remove = "_TwoPointCrossover_RadiusSamplingMutation_100_100000_RandomSampling_PathRepair"
# Loop through each item in the parent folder
for item in os.listdir(parent_folder_path):
    # Create the full path to the item
    old_folder_path = os.path.join(parent_folder_path, item)
    
    # Check if the item is a directory
    if os.path.isdir(old_folder_path):
        # Check if the part to remove is in the folder name
        if part_to_remove in item:
            # Create the new folder name by removing the specified part
            new_folder_name = item.replace(part_to_remove, "")
            # Create the full path for the new folder
            new_folder_path = os.path.join(parent_folder_path, new_folder_name)
            # Rename the folder
            os.rename(old_folder_path, new_folder_path)
            print(f"Renamed '{old_folder_path}' to '{new_folder_path}'")
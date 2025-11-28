import json
import os

# Define the path to the problematic map file
# Assuming this is the file the user wants to patch
MAP_PATH = "results/maps/llama_3b/fisher_gsm8k_mean.json"

def patch_map():
    """
    Converts the keys in the specified JSON map file from parameter names
    to module names by removing the '.weight' suffix. This patches old
    sensitivity maps to be compatible with the updated quantizer code.
    """
    if not os.path.exists(MAP_PATH):
        print(f"Error: The file was not found at '{MAP_PATH}'. Please ensure the path is correct.")
        return

    print(f"Reading map from: {MAP_PATH}")
    with open(MAP_PATH, 'r') as f:
        old_map = json.load(f)

    # Check if migration is needed by looking for at least one key ending in '.weight'
    if not any(key.endswith(".weight") for key in old_map.keys()):
        print("Map keys appear to be in the correct module-name format already. No patching needed.")
        return

    # Create the new map by renaming the keys
    new_map = {key.replace(".weight", ""): value for key, value in old_map.items()}

    # Overwrite the original file with the corrected map
    print(f"Patching file: converting keys from parameter names to module names...")
    with open(MAP_PATH, 'w') as f:
        json.dump(new_map, f, indent=2)

    print("Patch complete! Your Fisher map is now compatible with the updated code.")

if __name__ == "__main__":
    patch_map()

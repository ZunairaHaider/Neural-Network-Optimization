import json
import os
import glob

def merge_json_files(root_folder, output_file):
    results = []
    
    # Find all json files recursively
    json_files = glob.glob(os.path.join(root_folder, '**', '*.json'), recursive=True)
    
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    # Write the merged result
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Merged JSON saved to {output_file}")

# ---- Configure these ----
root_folder = "results/pruning"
output_file = "merged_results_pruning.json"
# -------------------------

merge_json_files(root_folder, output_file)
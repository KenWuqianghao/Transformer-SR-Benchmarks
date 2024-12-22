import json
import yaml
import os
import pandas as pd
import numpy as np

def convert_json_to_feynman(json_path, output_dir):
    """
    Convert symbolic regression JSON data to AI Feynman format
    
    Args:
        json_path: Path to input JSON file
        output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON data
    with open(json_path) as f:
        data = json.load(f)
    
    # Extract dataset name from json path
    dataset_name = os.path.splitext(os.path.basename(json_path))[0]
    
    # Create pandas dataframe from points data
    df_data = {
        **{f"var_{i}": data["points"][f"var_{i}"] for i in range(data["n_vars"])},
        "target": data["points"]["target"]
    }
    df = pd.DataFrame(df_data)
    
    # Save TSV file
    tsv_path = os.path.join(output_dir, f"{dataset_name}.tsv")
    df.to_csv(tsv_path, sep='\t', index=False)
    
    # Create metadata
    metadata = {
        "dataset": dataset_name,
        "description": "A synthetic symbolic regression model.\n"
                      "Formula and variable ranges given below.\n\n"
                      f"formula = {data['formula']}\n\n"
                      f"Variables: {data['n_vars']}\n"
                      f"Constants: {data['n_consts']}\n",
        "source": "Synthetic symbolic regression dataset",
        "task": "regression",
        "keywords": [
            "symbolic regression",
            "synthetic"
        ],
        "target": {
            "type": "continuous",
            "description": "Target variable"
        },
        "features": []
    }
    
    # Add feature metadata
    for i in range(data["n_vars"]):
        var_bounds = data["var_bound_dict"][f"var_{i}"]
        metadata["features"].append({
            "name": f"var_{i}",
            "type": "continuous", 
            "description": f"Input variable {i}",
            "bounds": var_bounds
        })
    
    # Add constants if any
    if data["const_value_dict"]:
        metadata["description"] += "\nConstants:\n"
        for const_name, const_value in data["const_value_dict"].items():
            metadata["description"] += f"{const_name} = {const_value}\n"
    
    # Save metadata YAML
    yaml_path = os.path.join(output_dir, "metadata.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    # Root directory for silver_fish
    silver_fish_dir = "silver_fish"
    
    # Walk through all directories in silver_fish
    for root, dirs, files in os.walk(silver_fish_dir):
        for file in files:
            if file.endswith('.json'):
                # Get full path to json file
                json_path = os.path.join(root, file)
                
                # Create corresponding output directory path
                rel_path = os.path.relpath(root, silver_fish_dir)
                dataset_name = os.path.splitext(file)[0]
                output_dir = os.path.join("filtered_silver_fish_datasets", rel_path, dataset_name)
                
                print(f"Converting {json_path} to {output_dir}")
                try:
                    convert_json_to_feynman(json_path, output_dir)
                except Exception as e:
                    print(f"Error converting {json_path}: {str(e)}") 
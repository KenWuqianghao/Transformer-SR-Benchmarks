import torch
import numpy as np
import sympy as sp
import os, sys
import symbolicregression
import requests
import pandas as pd
from sklearn.metrics import r2_score
from pathlib import Path

model_path = "model.pt" 
try:
    if not os.path.isfile(model_path): 
        url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
        r = requests.get(url, allow_redirects=True)
        open(model_path, 'wb').write(r.content)
    if not torch.cuda.is_available():
        model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(model_path)
        model = model.cuda()
    print(model.device)
    print("Model successfully loaded!")

except Exception as e:
    print("ERROR: model not loaded! path was: {}".format(model_path))
    print(e)    

est = symbolicregression.model.SymbolicTransformerRegressor(
                        model=model,
                        max_input_points=200,
                        n_trees_to_refine=50,
                        rescale=True                        
)

def get_equation(est):
    """Helper function to safely get equation from model"""
    try:
        tree_info = est.retrieve_tree()  # Removed with_infos=True
        replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
        model_str = tree_info.infix()
        for op, replace_op in replace_ops.items():
            model_str = model_str.replace(op, replace_op)
        return model_str
    except Exception as e:
        print(f"Error getting equation: {str(e)}")
        return "Unable to retrieve equation"

def calculate_r2_for_all_datasets(base_path='../filtered_black_box_datasets'):
    """
    Traverses all folders in base_path and calculates R2 scores for each dataset
    """
    results = {}
    
    for dataset_dir in Path(base_path).iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            tsv_path = dataset_dir / f"{dataset_name}.tsv"
            
            if tsv_path.exists():
                try:
                    # Read the TSV file
                    df = pd.read_csv(tsv_path, sep='\t')
                    
                    # Get feature columns (all columns except 'target')
                    feature_cols = [col for col in df.columns if col != 'target']
                    X = df[feature_cols].values
                    y = df['target'].values
                    
                    # Fit the symbolic regression model
                    est.fit(X, y)
                    
                    # Get predictions
                    y_pred = est.predict(X)
                    
                    # Calculate R2 score
                    r2 = r2_score(y, y_pred)
                    
                    # Get the equation
                    equation = get_equation(est)
                    
                    results[dataset_name] = {
                        'r2': r2,
                        'equation': equation
                    }
                    
                    print(f"Processed {dataset_name}: R2 = {r2:.4f}")
                    
                except Exception as e:
                    print(f"Error processing {dataset_name}: {str(e)}")
                    results[dataset_name] = None
    
    return results

# Run the function and display results
r2_scores = calculate_r2_for_all_datasets()

# Display results in a sorted manner
print("\nResults Summary:")
print("-" * 60)
for dataset, result in sorted(r2_scores.items()):
    if result is not None:
        print(f"\nDataset: {dataset}")
        print(f"R2 Score: {result['r2']:.4f}")
        print(f"Equation: {result['equation']}")
    else:
        print(f"\nDataset: {dataset}")
        print("Failed to process")
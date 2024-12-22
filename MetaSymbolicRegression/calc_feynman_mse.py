import torch
import numpy as np
import sympy as sp
import os, sys
import symbolicregression
import requests
import pandas as pd
from sklearn.metrics import r2_score
from pathlib import Path
import yaml
import math

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
    max_input_points=100,
    n_trees_to_refine=25,
    rescale=True
)

def get_equation(est):
    """Helper function to safely get equation from model"""
    try:
        tree_info = est.retrieve_tree()
        replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
        model_str = tree_info.infix()
        for op, replace_op in replace_ops.items():
            model_str = model_str.replace(op, replace_op)
        return model_str
    except Exception as e:
        print(f"Error getting equation: {str(e)}")
        return "Unable to retrieve equation"

def evaluate_formula(formula, feature_names, X):
    """
    Evaluates a mathematical formula using feature values
    """
    try:
        # Create a dictionary of sympy symbols for each feature
        symbols = {}
        for name in feature_names:
            # Handle variable names with underscores
            safe_name = name.replace('_', '__')
            symbols[name] = sp.Symbol(safe_name)
        
        # Add mathematical constants and functions
        math_symbols = {
            'pi': sp.pi,
            'exp': sp.exp,
            'sin': sp.sin,
            'cos': sp.cos,
            'sqrt': sp.sqrt,
            'arcsin': sp.asin,
            'arccos': sp.acos,
            'arctan': sp.atan,
        }
        symbols.update(math_symbols)
        
        # Clean up the formula
        formula = formula.replace('**', '^')  # Replace Python power operator with sympy power
        formula = formula.replace('math.pi', 'pi')  # Replace any remaining math.pi
        
        # Replace variable names in formula with safe names
        for name in feature_names:
            if '_' in name:
                safe_name = name.replace('_', '__')
                formula = formula.replace(name, safe_name)
        
        # Parse the formula
        expr = sp.sympify(formula, locals=symbols)
        
        # Convert back to using power operator that numpy understands
        expr = expr.replace(sp.Pow, lambda x, y: x**y)
        
        # Create a lambda function for fast evaluation
        variables = [symbols[name] for name in feature_names]
        f = sp.lambdify(variables, expr, modules=['numpy'])
        
        # Prepare inputs as a list of arrays in the correct order
        inputs = [X[:, feature_names.index(name)] for name in feature_names]
        
        # Evaluate the formula
        return f(*inputs)
        
    except Exception as e:
        print(f"Error evaluating formula: {str(e)}")
        print(f"Formula was: {formula}")
        print(f"Feature names: {feature_names}")
        raise

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def safe_mse(y_true, y_pred):
    """
    Calculate MSE in a numerically stable way
    """
    try:
        # Convert to float64 for better precision
        y_true = y_true.astype(np.float64)
        y_pred = y_pred.astype(np.float64)
        
        # Scale the values if they're too large
        max_val = max(np.max(np.abs(y_true)), np.max(np.abs(y_pred)))
        if max_val > 1e10:
            scale = max_val / 1e5
            y_true = y_true / scale
            y_pred = y_pred / scale
        
        # Calculate MSE
        squared_diff = np.square(y_true - y_pred)
        
        # Handle any remaining infinities or NaNs
        squared_diff = np.nan_to_num(squared_diff, nan=1e10, posinf=1e10, neginf=1e10)
        
        return np.mean(squared_diff)
    except Exception as e:
        print(f"Error in MSE calculation: {str(e)}")
        return float('inf')

def calculate_feynman_mse(base_path='../filtered_black_box_datasets'):
    """
    Traverses Feynman datasets and calculates MSE between true formula and predictions
    """
    results = {}
    
    for dataset_dir in Path(base_path).iterdir():
        if dataset_dir.is_dir():
            # Clear GPU memory before processing each dataset
            clear_gpu_memory()
            
            dataset_name = dataset_dir.name
            tsv_path = dataset_dir / f"{dataset_name}.tsv"
            metadata_path = dataset_dir / "metadata.yaml"
            
            if tsv_path.exists() and metadata_path.exists():
                try:
                    # Read the metadata and extract formula
                    with open(metadata_path, 'r') as f:
                        metadata = yaml.safe_load(f)
                        description = metadata['description']
                        
                        # Extract formula using the pattern of empty lines
                        lines = description.split('\n')
                        formula = None
                        for i, line in enumerate(lines):
                            if line.strip() == '':  # Found an empty line
                                # Check if next line contains formula and is followed by empty line
                                if (i + 1 < len(lines) and 
                                    i + 2 < len(lines) and 
                                    lines[i + 1].strip() != '' and 
                                    lines[i + 2].strip() == ''):
                                    # Extract only the right side of the equation
                                    formula_parts = lines[i + 1].strip().split('=')
                                    if len(formula_parts) == 2:
                                        formula = formula_parts[1].strip()
                                        # Replace mathematical constants
                                        formula = formula.replace('math.pi', 'pi')
                                        break
                        
                        if not formula:
                            print(f"No formula found in metadata for {dataset_name}")
                            continue
                    
                    # Read the data
                    df = pd.read_csv(tsv_path, sep='\t')
                    
                    # Get feature columns
                    feature_cols = [col for col in df.columns if col != 'target']
                    X = df[feature_cols].values
                    y_actual = df['target'].values
                    
                    # Sample exactly 10 points for each equation
                    if len(X) > 10:
                        indices = np.random.choice(len(X), 10, replace=False)
                        X = X[indices]
                        y_actual = y_actual[indices]
                    
                    # Calculate true values using formula from metadata
                    try:
                        y_true = evaluate_formula(formula, feature_cols, X)
                    except Exception as e:
                        print(f"Error evaluating formula for {dataset_name}: {str(e)}")
                        continue
                    
                    # Fit symbolic regression model
                    est.fit(X, y_actual)
                    y_pred = est.predict(X)
                    
                    # Clear GPU memory after processing
                    clear_gpu_memory()
                    
                    # Calculate MSE
                    mse_symbolic = safe_mse(y_actual, y_pred)
                    mse_true = safe_mse(y_actual, y_true)
                    
                    # Get the equation
                    equation = get_equation(est)
                    
                    results[dataset_name] = {
                        'mse_symbolic': mse_symbolic,
                        'mse_true': mse_true,
                        'equation': equation,
                        'true_formula': formula
                    }
                    
                    print(f"\nProcessed {dataset_name}:")
                    print(f"True formula: {formula}")
                    print(f"Symbolic MSE: {mse_symbolic:.6f}")
                    print(f"True Formula MSE: {mse_true:.6f}")
                    
                except Exception as e:
                    print(f"Error processing {dataset_name}: {str(e)}")
                    results[dataset_name] = None
    
    return results

def calculate_mse_statistics(mse_scores):
    """
    Calculate summary statistics for MSE scores
    """
    symbolic_mses = []
    true_mses = []
    
    for result in mse_scores.values():
        if result is not None:
            symbolic_mses.append(result['mse_symbolic'])
            true_mses.append(result['mse_true'])
    
    stats = {
        'symbolic': {
            'mean': np.mean(symbolic_mses),
            'median': np.median(symbolic_mses),
            'std': np.std(symbolic_mses),
            'min': np.min(symbolic_mses),
            'max': np.max(symbolic_mses)
        },
        'true': {
            'mean': np.mean(true_mses),
            'median': np.median(true_mses),
            'std': np.std(true_mses),
            'min': np.min(true_mses),
            'max': np.max(true_mses)
        },
        'total_datasets': len(mse_scores),
        'successful_datasets': len(symbolic_mses)
    }
    
    return stats

# Run the function and display results
mse_scores = calculate_feynman_mse()

# Calculate statistics
stats = calculate_mse_statistics(mse_scores)

# Display summary statistics
print("\nOverall Statistics:")
print("=" * 60)
print(f"Total datasets: {stats['total_datasets']}")
print(f"Successfully processed datasets: {stats['successful_datasets']}")
print("\nSymbolic Regression MSE Statistics:")
print("-" * 40)
print(f"Mean MSE: {stats['symbolic']['mean']:.6f}")
print(f"Median MSE: {stats['symbolic']['median']:.6f}")
print(f"Standard Deviation: {stats['symbolic']['std']:.6f}")
print(f"Min MSE: {stats['symbolic']['min']:.6f}")
print(f"Max MSE: {stats['symbolic']['max']:.6f}")

print("\nTrue Formula MSE Statistics:")
print("-" * 40)
print(f"Mean MSE: {stats['true']['mean']:.6f}")
print(f"Median MSE: {stats['true']['median']:.6f}")
print(f"Standard Deviation: {stats['true']['std']:.6f}")
print(f"Min MSE: {stats['true']['min']:.6f}")
print(f"Max MSE: {stats['true']['max']:.6f}")

# Display individual results
print("\nDetailed Results:")
print("-" * 60)
for dataset, result in sorted(mse_scores.items()):
    if result is not None:
        print(f"\nDataset: {dataset}")
        print(f"True formula: {result['true_formula']}")
        print(f"Symbolic MSE: {result['mse_symbolic']:.6f}")
        print(f"True Formula MSE: {result['mse_true']:.6f}")
        print(f"Discovered Equation: {result['equation']}")
    else:
        print(f"\nDataset: {dataset}")
        print("Failed to process")
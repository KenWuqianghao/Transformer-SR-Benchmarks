import torch
import numpy as np
import os, sys
import symbolicregression
import requests
import pandas as pd
from sklearn.metrics import r2_score
from pathlib import Path
from symbolicregression.model import SymbolicTransformerRegressor
import json
from datetime import datetime
import logging
import sys
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    # Your code here that generates the warning
sys.path.append('/home/k6wu/experiments/symbolicregression/')  # Add the path to where symbolicregression package is located

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

def create_estimator():
    """Create and configure the symbolic regression estimator"""
    
    est = SymbolicTransformerRegressor(
        model=model,
        max_input_points=100,
        n_trees_to_refine=25,
        rescale=True
    )
    return est

def safe_mse(y_true, y_pred):
    """Calculate MSE in a numerically stable way"""
    try:
        # Convert to float64 for better precision
        y_true = y_true.astype(np.float64)
        y_pred = y_pred.astype(np.float64)
        
        # Replace any infinity values
        y_true = np.nan_to_num(y_true, nan=1e10, posinf=1e10, neginf=-1e10)
        y_pred = np.nan_to_num(y_pred, nan=1e10, posinf=1e10, neginf=-1e10)
        
        # Scale the values if they're too large
        max_val = max(np.max(np.abs(y_true)), np.max(np.abs(y_pred)))
        if max_val > 1e10:
            scale = max_val / 1e5
            y_true = y_true / scale
            y_pred = y_pred / scale
        
        # Calculate MSE
        squared_diff = np.square(y_true - y_pred)
        squared_diff = np.nan_to_num(squared_diff, nan=1e10, posinf=1e10, neginf=1e10)
        return np.mean(squared_diff)
    except Exception as e:
        print(f"Error in MSE calculation: {str(e)}")
        return float('inf')

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

def calculate_metrics(tsv_path):
    """
    Calculate metrics for a single TSV file.
    
    Args:
        tsv_path (str): Path to the TSV file
        
    Returns:
        dict: Dictionary containing MSE, R² scores and the equation
    """
    try:
        # Read the data
        df = pd.read_csv(tsv_path, sep='\t')
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col != 'target']
        X = df[feature_cols].values
        y = df['target'].values
        
        # Replace any infinity values in the input data
        X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
        y = np.nan_to_num(y, nan=0, posinf=1e10, neginf=-1e10)
        
        # Sample 100 points if dataset is too large
        if len(X) > 100:
            indices = np.random.choice(len(X), 100, replace=False)
            X = X[indices]
            y = y[indices]
        
        # Create and fit symbolic regression model
        est = create_estimator()
        est.fit(X, y)
        y_pred = est.predict(X)
        
        # Calculate metrics
        mse = safe_mse(y, y_pred)
        r2 = r2_score(y, y_pred)
        equation = get_equation(est)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'mse': float(mse),
            'r2': float(r2),
            'equation': equation
        }
        
    except Exception as e:
        print(f"Error processing file {tsv_path}: {str(e)}")
        return None

def calculate_statistics(scores):
    """Calculate summary statistics for scores"""
    mse_scores = []
    r2_scores = []
    
    for result in scores.values():
        if result is not None:
            mse_scores.append(result['mse'])
            r2_scores.append(result['r2'])
    
    # Handle case where no successful results exist
    if not mse_scores or not r2_scores:
        return {
            'mse': {
                'mean': float('nan'),
                'median': float('nan'),
                'std': float('nan'),
                'min': float('nan'),
                'max': float('nan')
            },
            'r2': {
                'mean': float('nan'),
                'median': float('nan'),
                'std': float('nan'),
                'min': float('nan'),
                'max': float('nan')
            },
            'total_datasets': len(scores),
            'successful_datasets': 0
        }
    
    # Original statistics calculation for when we have results
    stats = {
        'mse': {
            'mean': np.mean(mse_scores),
            'median': np.median(mse_scores),
            'std': np.std(mse_scores),
            'min': np.min(mse_scores),
            'max': np.max(mse_scores)
        },
        'r2': {
            'mean': np.mean(r2_scores),
            'median': np.median(r2_scores),
            'std': np.std(r2_scores),
            'min': np.min(r2_scores),
            'max': np.max(r2_scores)
        },
        'total_datasets': len(scores),
        'successful_datasets': len(mse_scores)
    }
    
    return stats

def get_dataset_name(base_path):
    """Extract dataset name from path for file naming"""
    return Path(base_path).name.lower()

def setup_logging(base_path='../filtered_feynman_datasets'):
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_name = get_dataset_name(base_path)
    log_file = f'symbolic_regression_{dataset_name}_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def save_results(scores, stats, base_path='../filtered_feynman_datasets', timestamp=None):
    """Save results to JSON files"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    dataset_name = get_dataset_name(base_path)
    
    # Save detailed results
    results_file = f'results_{dataset_name}_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(scores, f, indent=4)
    
    # Save summary statistics
    stats_file = f'statistics_{dataset_name}_{timestamp}.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    return results_file, stats_file

# Modify the main execution part
if __name__ == "__main__":
    base_path = "./filtered_silver_fish_datasets/val/f_12/f_12.tsv"
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = setup_logging(base_path)
    logging.info("Starting symbolic regression metrics calculation")
    logging.info(f"Processing datasets from: {base_path}")
    
    try:
        # Run the calculations
        scores = calculate_metrics(base_path)
        print(scores)
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise

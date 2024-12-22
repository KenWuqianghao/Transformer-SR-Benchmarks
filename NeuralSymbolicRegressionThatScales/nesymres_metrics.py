import warnings

# Ignore specific warning messages
warnings.filterwarnings('ignore', message='Failed to initialize NumPy: _ARRAY_API not found')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from nesymres.architectures.model import Model
from nesymres.utils import load_metadata_hdf5
from nesymres.dclasses import FitParams, NNEquation, BFGSParams
from pathlib import Path
from functools import partial
import torch
from sympy import lambdify, sympify, symbols
import json
import numpy as np
import omegaconf
import pandas as pd
from sklearn.metrics import r2_score

def load_configurations(eq_setting_path: str, config_path: str):
    """Load equation and architecture configurations."""
    with open(eq_setting_path, 'r') as json_file:
        eq_setting = json.load(json_file)
    
    cfg = omegaconf.OmegaConf.load(config_path)
    return eq_setting, cfg

def setup_bfgs_params(cfg):
    """Set up BFGS parameters from config."""
    return BFGSParams(
        activated=cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )

def setup_fit_params(eq_setting, cfg, bfgs):
    """Set up fitting parameters."""
    return FitParams(
        word2id=eq_setting["word2id"],
        id2word={int(k): v for k,v in eq_setting["id2word"].items()},
        una_ops=eq_setting["una_ops"],
        bin_ops=eq_setting["bin_ops"],
        total_variables=list(eq_setting["total_variables"]),
        total_coefficients=list(eq_setting["total_coefficients"]),
        rewrite_functions=list(eq_setting["rewrite_functions"]),
        bfgs=bfgs,
        beam_size=cfg.inference.beam_size
    )

def load_model(weights_path: str, cfg):
    """Load and setup the model."""
    model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model

def generate_data(dataset_path, max_points=500):
    """Load data points from TSV files."""

    # Load data from TSV file
    df = pd.read_csv(dataset_path, sep='\t')
    
    # Number of variables is number of columns minus 1 (target column)
    n_variables = df.shape[1] - 1
    
    if n_variables > 3:
        return None, None
    
    # Trim data to max_points if necessary
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)  # Use random_state for reproducibility
        
    # Convert to tensors
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32)
    
    return X, y

def parse_and_evaluate_equation(equation_str, X):
    """
    Parse equation string and evaluate it with input data X
    
    Args:
        equation_str (str): Equation string like '((x_1)*(sin(x_1)))'
        X (numpy.ndarray): Input data with shape (n_samples, n_features)
        
    Returns:
        numpy.ndarray: Predicted values
    """
    # Remove the list brackets and clean the string
    equation_str = equation_str[0] if isinstance(equation_str, list) else equation_str
    
    # Create symbolic variables (x_1, x_2, etc.)
    var_symbols = {}
    for i in range(X.shape[1]):
        var_symbols[f'x_{i+1}'] = symbols(f'x_{i+1}')
    
    # Convert string to SymPy expression
    expr = sympify(equation_str)
    
    # Create a lambda function
    f = lambdify(list(var_symbols.values()), expr, modules=['numpy'])
    
    # Evaluate function
    if X.shape[1] == 1:
        y_pred = f(X[:, 0])
    else:
        y_pred = f(*[X[:, i] for i in range(X.shape[1])])
    
    return y_pred

def nesymres_evaluate(dataset_path):
    """
    Evaluate a single dataset using NeSyMReS model.
    
    Args:
        dataset_path (str): Path to the TSV file
        
    Returns:
        dict: Dictionary containing MSE, R², and equation
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Paths
            eq_setting_path = '/home/k6wu/experiments/NeuralSymbolicRegressionThatScales/jupyter/100M/eq_setting.json'
            config_path = '/home/k6wu/experiments/NeuralSymbolicRegressionThatScales/jupyter/100M/config.yaml'
            weights_path = '/home/k6wu/experiments/NeuralSymbolicRegressionThatScales/weights/100M.ckpt'
            
            # Load configurations
            eq_setting, cfg = load_configurations(eq_setting_path, config_path)
            
            # Setup parameters
            bfgs = setup_bfgs_params(cfg)
            params_fit = setup_fit_params(eq_setting, cfg, bfgs)
            
            # Load model
            model = load_model(weights_path, cfg)
            
            # Generate data
            X, y = generate_data(dataset_path)
            if X is None or y is None:
                return None
            
            # Get model predictions
            fitfunc = partial(model.fitfunc, cfg_params=params_fit)
            output = fitfunc(X, y)
            
            # Convert data to numpy
            X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
            y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
            
            # Get predictions
            equation_str = output['best_bfgs_preds']
            predictions = parse_and_evaluate_equation(equation_str, X_np)
            
            # Calculate metrics
            mse = np.mean((predictions - y_np) ** 2)
            r2 = r2_score(y_np, predictions)
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                'mse': float(mse),
                'r2': float(r2),
                'equation': equation_str
            }
            
    except Exception as e:
        print(f"Error processing {dataset_path}: {str(e)}")
        return None

def main():
    """Example usage of nesymres_evaluate"""
    # Dataset directories to process
    dataset_dirs = [
        "../filtered_black_box_datasets",
        "../filtered_feynman_datasets",
        "../filtered_silver_fish_datasets/test",
        "../filtered_silver_fish_datasets/train"
    ]
    
    results = {}
    
    # Process each dataset directory
    for base_dir in dataset_dirs:
        if not Path(base_dir).exists():
            print(f"Directory not found: {base_dir}")
            continue
            
        print(f"\nProcessing datasets in: {base_dir}")
        
        # Walk through all directories
        for dataset_dir in Path(base_dir).iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                tsv_path = dataset_dir / f"{dataset_name}.tsv"
                
                if tsv_path.exists():
                    print(f"\nProcessing {dataset_name}...")
                    metrics = nesymres_evaluate(tsv_path)
                    
                    if metrics:
                        results[dataset_name] = metrics
                        print(f"MSE: {metrics['mse']:.6f}")
                        print(f"R² Score: {metrics['r2']:.6f}")
                        print(f"Equation: {metrics['equation']}")
    
    # Calculate aggregate statistics
    successful_datasets = [m for m in results.values() if m is not None]
    if successful_datasets:
        mse_values = [m['mse'] for m in successful_datasets]
        r2_values = [m['r2'] for m in successful_datasets]
        
        print("\nOverall Statistics:")
        print("=" * 50)
        print(f"Total datasets processed: {len(results)}")
        print(f"Successfully processed: {len(successful_datasets)}")
        print("\nMSE Statistics:")
        print(f"Mean: {np.mean(mse_values):.6f}")
        print(f"Median: {np.median(mse_values):.6f}")
        print(f"Std Dev: {np.std(mse_values):.6f}")
        print("\nR² Statistics:")
        print(f"Mean: {np.mean(r2_values):.6f}")
        print(f"Median: {np.median(r2_values):.6f}")
        print(f"Std Dev: {np.std(r2_values):.6f}")
    
    return results

if __name__ == "__main__":
    main()

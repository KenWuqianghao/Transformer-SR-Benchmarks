import warnings
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging
import torch
import os
import sys
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*functorch.*')

# Add all repository paths to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(current_dir, 'NeuralSymbolicRegressionThatScales'),
    os.path.join(current_dir, 'TPSR'),
    os.path.join(current_dir, 'symbolicregression')
])

# Import evaluation functions from each model
from NeuralSymbolicRegressionThatScales.nesymres_metrics import nesymres_evaluate
from MetaSymbolicRegression.meta_metrics import calculate_metrics as symbolic_evaluate
from TPSR.tpsr_metrics import evaluate_tsv_file as tpsr_evaluate
from convert_to_pmlb import convert_json_to_feynman

def setup_logging(output_dir):
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'model_comparison_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def evaluate_single_file(tsv_path):
    """
    Evaluate a single TSV file using all three models
    
    Args:
        tsv_path (str): Path to the TSV file
        
    Returns:
        dict: Results from all models
    """
    results = {
        'dataset': Path(tsv_path).stem,
        'nesymres': None,
        'symbolic': None,
        'tpsr': None
    }
    
    logging.info(f"\nEvaluating file: {tsv_path}")
    
    # Evaluate NeSyMReS model
    try:
        results['nesymres'] = nesymres_evaluate(tsv_path)
        logging.info("NeSyMReS evaluation successful")
    except Exception as e:
        logging.error(f"NeSyMReS evaluation failed: {str(e)}")
    
    # Evaluate Symbolic Regression model
    try:
        results['symbolic'] = symbolic_evaluate(tsv_path)
        logging.info("Symbolic regression evaluation successful")
    except Exception as e:
        logging.error(f"Symbolic regression evaluation failed: {str(e)}")
    
    # Evaluate TPSR model
    try:
        results['tpsr'] = tpsr_evaluate(tsv_path, json_dir, results_csv, target_formula)
        logging.info("TPSR evaluation successful")
    except Exception as e:
        logging.error(f"TPSR evaluation failed: {str(e)}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

def generate_overall_statistics(results_csv, output_dir):
    """
    Generate overall statistics for each model and data split
    
    Args:
        results_csv (str): Path to the CSV file containing all results
        output_dir (str): Directory to save the statistics
    """
    # Read the results
    df = pd.read_csv(results_csv)
    
    # Get unique models
    models = df['model'].unique()
    
    # Initialize statistics dictionary
    stats = {split: {model: {} for model in models} for split in ['train', 'test', 'val']}
    
    # Calculate statistics for each model and split
    for split in ['train', 'test', 'val']:
        split_df = df[df['dataset'].str.contains(f'/{split}/')]
        
        for model in models:
            model_df = split_df[split_df['model'] == model]
            
            if not model_df.empty:
                # Calculate MSE statistics
                mse_stats = {
                    'mse_median': model_df['mean_squared_error'].median(),
                    'mse_mean': model_df['mean_squared_error'].mean(),
                    'mse_max': model_df['mean_squared_error'].max(),
                    'mse_min': model_df['mean_squared_error'].min(),
                    'mse_std': model_df['mean_squared_error'].std()
                }
                
                # Calculate R2 statistics
                r2_stats = {
                    'r2_median': model_df['R2_score'].median(),
                    'r2_mean': model_df['R2_score'].mean(),
                    'r2_max': model_df['R2_score'].max(),
                    'r2_min': model_df['R2_score'].min(),
                    'r2_std': model_df['R2_score'].std()
                }
                
                # Add number of successful evaluations
                eval_stats = {
                    'num_evaluations': len(model_df),
                    'num_successful_predictions': model_df['prediction_human_form'].notna().sum()
                }
                
                # Combine all statistics
                stats[split][model] = {**mse_stats, **r2_stats, **eval_stats}
    
    # Create a more readable format for the statistics file
    formatted_stats = []
    for split in stats:
        for model in stats[split]:
            row = {
                'split': split,
                'model': model,
                **stats[split][model]
            }
            formatted_stats.append(row)
    
    # Convert to DataFrame for easier viewing
    stats_df = pd.DataFrame(formatted_stats)
    
    # Save statistics
    stats_file = os.path.join(output_dir, 'overall_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    
    # Also save as a more detailed JSON
    json_stats_file = os.path.join(output_dir, 'overall_statistics.json')
    with open(json_stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    return stats

def evaluate_directory(json_dir, output_dir='evaluation_results'):
    """
    Evaluate all JSON files in a directory by first converting them to TSV format
    and then running model evaluations.
    """
    # Create output directories
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(output_dir)
    logging.info(f"Starting model comparison on directory: {json_dir}")
    logging.info(f"Results will be saved in: {results_dir}")
    logging.info(f"Logs will be saved in: {os.path.dirname(log_file)}")
    
    # Define CSV file path for all results
    results_csv = os.path.join(results_dir, 'model_comparison_results.csv')
    
    try:
        # Walk through all JSON files
        for root, _, files in os.walk(json_dir):
            for file in files:
                if file.endswith('.json') and file != 'properties.json':
                    json_path = os.path.join(root, file)
                    dataset_name = os.path.splitext(file)[0]
                    
                    # Get target formula
                    try:
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            target_formula = data.get('formula', None)
                    except:
                        target_formula = None
                    
                    # Create TSV directory path
                    rel_path = os.path.relpath(root, json_dir)
                    tsv_dir = os.path.join('filtered_datasets', rel_path, dataset_name)
                    
                    logging.info(f"\nConverting {json_path} to TSV format...")
                    try:
                        convert_json_to_feynman(json_path, tsv_dir)
                        tsv_path = os.path.join(tsv_dir, f"{dataset_name}.tsv")
                        
                        # Evaluate using each model
                        try:
                            # NeSyMReS evaluation
                            try:
                                nesymres_results = nesymres_evaluate(tsv_path)
                                logging.info("NeSyMReS evaluation successful")
                                # Save NeSyMReS results
                                # Extract formula from list if needed
                                prediction = nesymres_results.get('equation', None)
                                if isinstance(prediction, list):
                                    prediction = prediction[0] if prediction else None
                                
                                new_row = pd.DataFrame([{
                                    'dataset': dataset_name,
                                    'model': 'nesymres',
                                    'target_human_form': target_formula,
                                    'prediction_human_form': prediction,
                                    'mean_squared_error': nesymres_results['mse'],
                                    'R2_score': nesymres_results['r2']
                                }])
                                if os.path.exists(results_csv):
                                    new_row.to_csv(results_csv, mode='a', header=False, index=False)
                                else:
                                    new_row.to_csv(results_csv, index=False)
                            except Exception as e:
                                logging.error(f"NeSyMReS evaluation failed: {str(e)}")
                            
                            # Symbolic regression evaluation
                            try:
                                symbolic_results = symbolic_evaluate(tsv_path)
                                logging.info("Symbolic regression evaluation successful")
                                # Save Symbolic results
                                new_row = pd.DataFrame([{
                                    'dataset': dataset_name,
                                    'model': 'symbolic',
                                    'target_human_form': target_formula,
                                    'prediction_human_form': symbolic_results.get('equation', None),
                                    'mean_squared_error': symbolic_results['mse'],
                                    'R2_score': symbolic_results['r2']
                                }])
                                new_row.to_csv(results_csv, mode='a', header=False, index=False)
                            except Exception as e:
                                logging.error(f"Symbolic regression evaluation failed: {str(e)}")
                            
                            # TPSR evaluation
                            try:
                                tpsr_results = tpsr_evaluate(tsv_path, json_dir, results_csv, target_formula)
                                logging.info("TPSR evaluation successful")
                            except Exception as e:
                                logging.error(f"TPSR evaluation failed: {str(e)}")

                            # Log results
                            logging.info(f"\nResults for {dataset_name}:")
                            if nesymres_results:
                                logging.info(f"NeSyMReS - MSE: {nesymres_results['mse']:.6f}, R²: {nesymres_results['r2']:.6f}")
                            if symbolic_results:
                                logging.info(f"Symbolic - MSE: {symbolic_results['mse']:.6f}, R²: {symbolic_results['r2']:.6f}")
                            if tpsr_results:
                                logging.info(f"TPSR - MSE: {tpsr_results['mse']:.6f}, R²: {tpsr_results['r2']:.6f}")
                                logging.info(f"TPSR Equation: {tpsr_results['equation']}")

                        except Exception as e:
                            logging.error(f"Error in model evaluation: {str(e)}")
                            continue

                    except Exception as e:
                        logging.error(f"Error processing {json_path}: {str(e)}")
                        continue

        logging.info(f"\nAll results saved to: {results_csv}")
        
        # Generate overall statistics
        logging.info("\nGenerating overall statistics...")
        stats = generate_overall_statistics(results_csv, results_dir)
        
        # Log summary statistics
        logging.info("\nSummary Statistics:")
        logging.info("=" * 60)
        for split in ['train', 'test', 'val']:
            logging.info(f"\n{split.upper()} SPLIT:")
            for model in stats[split]:
                logging.info(f"\n{model.upper()}:")
                logging.info(f"MSE - Median: {stats[split][model]['mse_median']:.6f}, "
                           f"Mean: {stats[split][model]['mse_mean']:.6f}, "
                           f"Min: {stats[split][model]['mse_min']:.6f}, "
                           f"Max: {stats[split][model]['mse_max']:.6f}")
                logging.info(f"R² - Median: {stats[split][model]['r2_median']:.6f}, "
                           f"Mean: {stats[split][model]['r2_mean']:.6f}, "
                           f"Min: {stats[split][model]['r2_min']:.6f}, "
                           f"Max: {stats[split][model]['r2_max']:.6f}")
                logging.info(f"Successful predictions: {stats[split][model]['num_successful_predictions']} "
                           f"out of {stats[split][model]['num_evaluations']}")

        logging.info(f"\nDetailed statistics saved to: {os.path.join(results_dir, 'overall_statistics.csv')}")
        logging.info(f"JSON format saved to: {os.path.join(results_dir, 'overall_statistics.json')}")
        
    except Exception as e:
        logging.error(f"Error in directory evaluation: {str(e)}", exc_info=True)
        raise

def main():
    """
    Main function to evaluate models on a directory of JSON files
    """
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate models on symbolic regression datasets')
    parser.add_argument('--json_dir', type=str, default='/home/k6wu/experiments/silver_fish',
                      help='Directory containing JSON files (default: silver_fish)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save results (default: evaluation_results)')
    args = parser.parse_args()
    
    evaluate_directory(args.json_dir, args.output_dir)

if __name__ == "__main__":
    main() 
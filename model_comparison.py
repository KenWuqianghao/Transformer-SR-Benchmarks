import warnings
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging
import torch
import os
import sys

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
        results['tpsr'] = tpsr_evaluate(tsv_path)
        logging.info("TPSR evaluation successful")
    except Exception as e:
        logging.error(f"TPSR evaluation failed: {str(e)}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

def evaluate_directory(json_dir, output_dir='evaluation_results'):
    """
    Evaluate all JSON files in a directory by first converting them to TSV format
    and then running model evaluations.
    
    Args:
        json_dir (str): Directory containing JSON files
        output_dir (str): Base directory for outputs
    
    Returns:
        dict: Results for all datasets
    """
    # Create output directories
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(output_dir)
    logging.info(f"Starting model comparison on directory: {json_dir}")
    logging.info(f"Results will be saved in: {results_dir}")
    logging.info(f"Logs will be saved in: {os.path.dirname(log_file)}")
    
    all_results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # Walk through all JSON files
        for root, _, files in os.walk(json_dir):
            for file in files:
                # Skip properties.json files
                if file.endswith('.json') and file != 'properties.json':
                    json_path = os.path.join(root, file)
                    dataset_name = os.path.splitext(file)[0]
                    
                    # Create TSV directory path
                    rel_path = os.path.relpath(root, json_dir)
                    tsv_dir = os.path.join('filtered_datasets', rel_path, dataset_name)
                    
                    # Convert JSON to TSV
                    logging.info(f"\nConverting {json_path} to TSV format...")
                    try:
                        convert_json_to_feynman(json_path, tsv_dir)
                        tsv_path = os.path.join(tsv_dir, f"{dataset_name}.tsv")
                        
                        # Evaluate the TSV file
                        results = evaluate_single_file(tsv_path)
                        all_results[dataset_name] = results
                        
                        # Save individual result
                        individual_result_file = os.path.join(results_dir, f'{dataset_name}_{timestamp}.json')
                        with open(individual_result_file, 'w') as f:
                            json.dump(results, f, indent=4)
                        
                        # Log individual results
                        logging.info(f"\nResults for {dataset_name}:")
                        for model in ['nesymres', 'symbolic', 'tpsr']:
                            if results[model]:
                                logging.info(f"\n{model.upper()} Results:")
                                logging.info(f"MSE: {results[model]['mse']:.6f}")
                                logging.info(f"R²: {results[model]['r2']:.6f}")
                                if 'equation' in results[model]:
                                    logging.info(f"Equation: {results[model]['equation']}")
                            else:
                                logging.info(f"\n{model.upper()}: Evaluation failed")
                                
                    except Exception as e:
                        logging.error(f"Error processing {json_path}: {str(e)}")
                        continue
        
        # Calculate and log aggregate statistics
        logging.info("\nAggregate Statistics:")
        logging.info("=" * 60)
        
        model_stats = {}
        for model in ['nesymres', 'symbolic', 'tpsr']:
            mse_values = []
            r2_values = []
            
            for result in all_results.values():
                if result[model]:
                    mse_values.append(result[model]['mse'])
                    r2_values.append(result[model]['r2'])
            
            if mse_values:
                model_stats[model] = {
                    'mse': {
                        'mean': float(np.mean(mse_values)),
                        'median': float(np.median(mse_values)),
                        'std': float(np.std(mse_values))
                    },
                    'r2': {
                        'mean': float(np.mean(r2_values)),
                        'median': float(np.median(r2_values)),
                        'std': float(np.std(r2_values))
                    },
                    'successful_evaluations': len(mse_values)
                }
                
                logging.info(f"\n{model.upper()}:")
                logging.info(f"Successful evaluations: {len(mse_values)}")
                logging.info(f"MSE - Mean: {np.mean(mse_values):.6f}, Median: {np.median(mse_values):.6f}, Std: {np.std(mse_values):.6f}")
                logging.info(f"R² - Mean: {np.mean(r2_values):.6f}, Median: {np.median(r2_values):.6f}, Std: {np.std(r2_values):.6f}")
        
        # Save aggregate results
        summary_file = os.path.join(results_dir, f'summary_{timestamp}.json')
        with open(summary_file, 'w') as f:
            json.dump({
                'aggregate_statistics': model_stats,
                'dataset_count': len(all_results)
            }, f, indent=4)
        
        # Save complete results
        complete_results_file = os.path.join(results_dir, f'complete_results_{timestamp}.json')
        with open(complete_results_file, 'w') as f:
            json.dump({
                'individual_results': all_results,
                'aggregate_statistics': model_stats,
                'dataset_count': len(all_results)
            }, f, indent=4)
        
        logging.info(f"\nIndividual results saved in: {results_dir}")
        logging.info(f"Summary results saved to: {summary_file}")
        logging.info(f"Complete results saved to: {complete_results_file}")
        logging.info(f"Full log available at: {log_file}")
        
        return all_results
        
    except Exception as e:
        logging.error(f"Error in directory evaluation: {str(e)}", exc_info=True)
        raise

def main():
    """
    Main function to evaluate models on a directory of JSON files
    """
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate models on symbolic regression datasets')
    parser.add_argument('--json_dir', type=str, default='silver_fish',
                      help='Directory containing JSON files (default: silver_fish)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save results (default: evaluation_results)')
    args = parser.parse_args()
    
    evaluate_directory(args.json_dir, args.output_dir)

if __name__ == "__main__":
    main() 
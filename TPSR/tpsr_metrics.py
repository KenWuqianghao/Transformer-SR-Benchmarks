import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import torch
import time
import json
import os
import warnings

# Suppress FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Suppress all warnings from torch.func deprecation
warnings.filterwarnings('ignore', message='.*functorch.*')

import torch
import numpy as np
import sympy as sp
from parsers import get_parser
from argparse import Namespace

import symbolicregression
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.trainer import Trainer
from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor , get_top_k_features
from symbolicregression.e2e_model import Transformer, pred_for_sample_no_refine, respond_to_batch , pred_for_sample, refine_for_sample, pred_for_sample_test, refine_for_sample_test 
from dyna_gym.agents.uct import UCT
from dyna_gym.agents.mcts import update_root, convert_to_json, print_tree
from rl_env import RLEnv
from default_pi import E2EHeuristic, NesymresHeuristic
from symbolicregression.metrics import compute_metrics
from symbolicregression.model.model_wrapper import ModelWrapper

from nesymres.src.nesymres.architectures.model import Model
from nesymres.utils import load_metadata_hdf5
from nesymres.dclasses import FitParams, NNEquation, BFGSParams
from functools import partial
from sympy import lambdify
from reward import compute_reward_e2e, compute_reward_nesymres
import omegaconf

def load_tsv_data(file_path):
    """Load data from TSV file and separate features and target"""
    df = pd.read_csv(file_path, sep='\t')
    X = df.filter(regex='^var_').values
    y = df['target'].values
    return X, y

def calculate_metrics(y_true, y_pred):
    """Calculate R² and MSE metrics"""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        'r2': r2,
        'mse': mse
    }

def evaluate_model_predictions(model_type, model, X, y_true, params=None, equation_env=None):
    """Evaluate model predictions using different methods based on model type"""
    if model_type == 'e2e':
        # Format data as expected by E2E model
        samples = {
            'x_to_fit': [X],
            'y_to_fit': [y_true.reshape(-1, 1)],
            'x_to_pred': [X],
            'y_to_pred': [y_true.reshape(-1, 1)]
        }
        
        # Create RL environment and default policy
        rl_env = RLEnv(
            params=params,
            samples=samples,
            equation_env=equation_env,
            model=model
        )

        dp = E2EHeuristic(
            equation_env=equation_env,
            rl_env=rl_env,
            model=model,
            k=params.width,
            num_beams=params.num_beams,
            horizon=params.horizon,
            device=params.device,
            use_seq_cache=not params.no_seq_cache,
            use_prefix_cache=not params.no_prefix_cache,
            length_penalty=params.beam_length_penalty,
            train_value_mode=params.train_value,
            debug=params.debug
        )

        # Initialize UCT agent
        agent = UCT(
            action_space=[],
            gamma=1.,
            ucb_constant=params.ucb_constant,
            horizon=params.horizon,
            rollouts=params.rollout,
            dp=dp,
            width=params.width,
            reuse_tree=True,
            alg=params.uct_alg,
            ucb_base=params.ucb_base
        )

        # Run MCTS
        horizon = 1 if params.sample_only else 200
        done = False
        s = rl_env.state
        
        for t in range(horizon):
            if len(s) >= params.horizon:
                break
            if done:
                break
                
            act = agent.act(rl_env, done)
            s, r, done, _ = rl_env.step(act)
            update_root(agent, act, s)
            dp.update_cache(s)
        
        # Get predictions and equation using refine_for_sample_test
        y_mcts_refine_train, y_mcts_refine_test, model_str, tree = refine_for_sample_test(
            model,
            equation_env, 
            s,  
            samples['x_to_fit'], 
            samples['y_to_fit'],
            samples['x_to_pred']
        )
        
        # Format the equation string
        replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
        for op, replace_op in replace_ops.items():
            if model_str:
                model_str = model_str.replace(op, replace_op)
        
        # Parse equation if possible
        try:
            if model_str:
                equation = sp.parse_expr(model_str)
            else:
                equation = None
        except:
            equation = None
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_mcts_refine_test)
        r2 = r2_score(y_true, y_mcts_refine_test)
        
        return {
            'mse': mse, 
            'r2': r2,
            'equation': str(equation) if equation else None
        }
    
    elif model_type == 'nesymres':
        # NeSymReS implementation remains unchanged
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_true, dtype=torch.float32, device=device)
        y_pred = model.predict(X_tensor).detach().cpu().numpy()
        metrics = calculate_metrics(y_true, y_pred)
        return metrics
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_default_params():
    """Get default parameters for E2E model evaluation"""
    params = Namespace(backbone_model='e2e', seed=23, width=3, horizon=200, rollout=3, num_beams=1, train_value=False, no_seq_cache=True, no_prefix_cache=True, sample_only=False, ucb_constant=1.0, ucb_base=10.0, uct_alg='uct', ts_mode='best', alg='mcts', entropy_weighted_strategy='none', dump_path='', refinements_types='method=BFGS_batchsize=256_metric=/_mse', eval_dump_path=None, save_results=True, exp_name='debug', print_freq=100, save_periodic=25, exp_id='', fp16=False, amp=-1, rescale=True, embedder_type='LinearPoint', emb_emb_dim=64, enc_emb_dim=512, dec_emb_dim=512, n_emb_layers=1, n_enc_layers=2, n_dec_layers=16, n_enc_heads=16, n_dec_heads=16, emb_expansion_factor=1, n_enc_hidden_layers=1, n_dec_hidden_layers=1, norm_attention=False, dropout=0, attention_dropout=0, share_inout_emb=True, enc_positional_embeddings=None, dec_positional_embeddings='learnable', env_base_seed=0, test_env_seed=1, batch_size=1, batch_size_eval=64, optimizer='adam_inverse_sqrt,warmup_updates=10000', lr=1e-05, clip_grad_norm=0.5, n_steps_per_epoch=3000, max_epoch=100000, stopping_criterion='', accumulate_gradients=1, num_workers=10, train_noise_gamma=0.0, ablation_to_keep=None, max_input_points=200, n_trees_to_refine=10, export_data=False, reload_data='', reload_size=-1, batch_load=False, env_name='functions', queue_strategy=None, collate_queue_size=2000, use_sympy=False, simplify=False, use_abs=False, operators_to_downsample='div_0,arcsin_0,arccos_0,tan_0.2,arctan_0.2,sqrt_5,pow2_3,inv_3', operators_to_not_repeat='', max_unary_depth=6, required_operators='', extra_unary_operators='', extra_binary_operators='', extra_constants=None, min_input_dimension=1, max_input_dimension=10, min_output_dimension=1, max_output_dimension=1, enforce_dim=True, use_controller=True, float_precision=3, mantissa_len=1, max_exponent=100, max_exponent_prefactor=1, max_token_len=0, tokens_per_batch=10000, pad_to_max_dim=True, max_int=10, min_binary_ops_per_dim=0, max_binary_ops_per_dim=1, max_binary_ops_offset=4, min_unary_ops=0, max_unary_ops=4, min_op_prob=0.01, max_len=200, min_len_per_dim=5, max_centroids=10, prob_const=0.0, reduce_num_constants=True, use_skeleton=False, prob_rand=0.0, max_trials=1, n_prediction_points=200, tasks='functions', beam_eval=True, max_generated_output_len=200, beam_eval_train=0, beam_size=1, beam_type='sampling', beam_temperature=0.1, beam_length_penalty=1, lam=0.1, beam_early_stopping=True, beam_selection_metrics=1, max_number_bags=10, reload_model='', reload_checkpoint='', validation_metrics='r2_zero,r2,accuracy_l1_biggio,accuracy_l1_1e-3,accuracy_l1_1e-2,accuracy_l1_1e-1,_complexity', debug_train_statistics=False, eval_noise_gamma=0.0, eval_size=10000, eval_noise_type='additive', eval_noise=0, eval_only=False, eval_from_exp='', eval_data='', eval_verbose=0, eval_verbose_print=False, eval_input_length_modulo=-1, eval_on_pmlb=False, eval_mcts_on_pmlb=False, eval_in_domain=False, eval_mcts_in_domain=False, debug_slurm=False, debug=True, cpu=False, local_rank=-1, gpu_to_use='0', master_port=-1, windows=False, nvidia_apex=False, max_src_len=200, max_target_len=200, reward_type='nmse', reward_coef=1, vf_coef=0.0001, target_kl=1, entropy_coef=0.01, kl_regularizer=0.001, warmup_epoch=5, lr_patience=100, save_model=True, save_eval_dic='./eval_result', update_modules='all', actor_lr=1e-06, critic_lr=1e-05, kl_coef=0.01, rl_alg='ppo', run_id=1, pmlb_data_type='feynman', target_noise=0.0, prediction_sigmas='1,2,4,8,16')
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    params.debug = True
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return params

def load_target_formula(data_dir, dataset_name):
    """Load the original formula from the dataset's JSON file"""
    json_path = os.path.join(data_dir, dataset_name, 'data.json')
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            return data.get('formula', None)
    except:
        return None

def evaluate_tsv_file(file_path, data_dir, csv_file):
    """
    Evaluate a TSV file and save results immediately to CSV
    
    Args:
        file_path (str): Path to the TSV file
        data_dir (str): Directory containing the original JSON files
        csv_file (str): Path to the CSV file to save/append results
    """
    # Get dataset name from file path
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Get target formula
    target_formula = load_target_formula(data_dir, dataset_name)
    
    # Load and evaluate data
    X, y_true = load_tsv_data(file_path)
    
    # Initialize model and get predictions
    params = get_default_params()
    equation_env = build_env(params)
    build_modules(equation_env, params)
    
    samples = {
        'x_to_fit': [X],
        'y_to_fit': [y_true.reshape(-1,1)],
        'x_to_pred': [X],
        'y_to_pred': [y_true.reshape(-1,1)]
    }
    
    if not params.cpu:
        assert torch.cuda.is_available()
    symbolicregression.utils.CUDA = not params.cpu
    
    model = Transformer(params=params, env=equation_env, samples=samples)
    model.to(params.device)
    
    # Get metrics and equation
    results = evaluate_model_predictions(
        model_type='e2e',
        model=model,
        X=X,
        y_true=y_true,
        params=params,
        equation_env=equation_env
    )
    
    # Create new row
    new_row = pd.DataFrame([{
        'dataset': dataset_name,
        'model': 'tpsr',
        'target_human_form': target_formula,
        'prediction_human_form': results['equation'],
        'mean_squared_error': results['mse'],
        'R2_score': results['r2']
    }])
    
    # Save/append to CSV
    if os.path.exists(csv_file):
        new_row.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        new_row.to_csv(csv_file, index=False)
    
    return results

def main():
    # Directory containing TSV files
    data_dir = "/home/k6wu/experiments/filtered_silver_fish_datasets/val"
    output_dir = "/home/k6wu/experiments/TPSR/results"
    
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define CSV file path
    csv_file = os.path.join(output_dir, "tpsr_results.csv")
    
    # Process each TSV file
    for subdir in os.listdir(data_dir):
        tsv_path = os.path.join(data_dir, subdir, f"{subdir}.tsv")
        if os.path.exists(tsv_path):
            print(f"\nProcessing {subdir}...")
            try:
                results = evaluate_tsv_file(tsv_path, data_dir, csv_file)
                
                # Print progress
                print(f"Dataset: {subdir}")
                print(f"Target Formula: {results.get('target_formula')}")
                print(f"Predicted Formula: {results['equation']}")
                print(f"R² Score: {results['r2']:.4f}")
                print(f"MSE: {results['mse']:.4f}")
            except Exception as e:
                print(f"Error processing {subdir}: {str(e)}")
    
    print(f"\nResults saved to: {csv_file}")

if __name__ == "__main__":
    main() 
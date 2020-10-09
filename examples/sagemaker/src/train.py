import ast
import argparse
import logging
import warnings
import os
import json
import glob
import subprocess
import sys
import boto3
import pickle
import pandas as pd
from collections import Counter
from timeit import default_timer as timer

sys.path.insert(0, 'package')
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from prettytable import PrettyTable
    import autogluon as ag
    from autogluon import TabularPrediction as task
    from autogluon.task.tabular_prediction import TabularDataset
    

# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #

def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    return subprocess.check_output(['du','-sh', path]).split()[0].decode('utf-8')

def __load_input_data(path: str) -> TabularDataset:
    """
    Load training data as dataframe
    :param path:
    :return: DataFrame
    """
    input_data_files = os.listdir(path)
    try:
        input_dfs = [pd.read_csv(f'{path}/{data_file}') for data_file in input_data_files]
        return task.Dataset(df=pd.concat(input_dfs))
    except:
        print(f'No csv data in {path}!')
        return None

def train(args):

    is_distributed = len(args.hosts) > 1
    host_rank = args.hosts.index(args.current_host)    
    dist_ip_addrs = args.hosts
    dist_ip_addrs.pop(host_rank)
    ngpus_per_trial = 1 if args.num_gpus > 0 else 0

    # load training and validation data
    print(f'Train files: {os.listdir(args.train)}')
    train_data = __load_input_data(args.train)
    print(f'Label counts: {dict(Counter(train_data[args.label]))}')
    print(f'hp: {args.hyperparameters}')
    
    predictor = task.fit(
        train_data=train_data,
        label=args.label,            
        output_directory=args.model_dir,
        problem_type=args.problem_type,
        eval_metric=args.eval_metric,
        stopping_metric=args.stopping_metric,
        auto_stack=args.auto_stack, # default: False
        hyperparameter_tune=args.hyperparameter_tune, # default: False
        feature_prune=args.feature_prune, # default: False
        holdout_frac=args.holdout_frac, # default: None
        num_bagging_folds=args.num_bagging_folds, # default: 0
        num_bagging_sets=args.num_bagging_sets, # default: None
        stack_ensemble_levels=args.stack_ensemble_levels, # default: 0
        hyperparameters=args.hyperparameters,
        cache_data=args.cache_data,
        time_limits=args.time_limits,
        num_trials=args.num_trials, # default: None
        search_strategy=args.search_strategy, # default: 'random'
        search_options=args.search_options,
        visualizer=args.visualizer,
        verbosity=args.verbosity
    )
    
    # Results summary
    predictor.fit_summary(verbosity=1)

    # Leaderboard on optional test data
    if args.test:
        print(f'Test files: {os.listdir(args.test)}')
        test_data = __load_input_data(args.test)    
        print('Running model on test data and getting Leaderboard...')
        leaderboard = predictor.leaderboard(dataset=test_data, silent=True)
        def format_for_print(df):
            table = PrettyTable(list(df.columns))
            for row in df.itertuples():
                table.add_row(row[1:])
            return str(table)
        print(format_for_print(leaderboard), end='\n\n')

    # Files summary
    print(f'Model export summary:')
    print(f"/opt/ml/model/: {os.listdir('/opt/ml/model/')}")
    models_contents = os.listdir('/opt/ml/model/models')
    print(f"/opt/ml/model/models: {models_contents}")
    print(f"/opt/ml/model directory size: {du('/opt/ml/model/')}\n")

# ------------------------------------------------------------ #
# Training execution                                           #
# ------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type','bool', lambda v: v.lower() in ('yes', 'true', 't', '1'))

    # Environment parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    # Arguments to be passed to task.fit()
    parser.add_argument('--fit_args', type=lambda s: ast.literal_eval(s),
                        default="{'presets': ['optimize_for_deployment']}",
                        help='https://autogluon.mxnet.io/api/autogluon.task.html#tabularprediction')
    # Additional options
    parser.add_argument('--feature_importance', type='bool', default=True)

    return parser.parse_args()


if __name__ == "__main__":
    start = timer()

    args = parse_args()
    
    # Print SageMaker args
    print('\n====== args ======')
    for k,v in vars(args).items():
        print(f'{k},  type: {type(v)},  value: {v}')
    print()
    
    # Convert AutoGluon hyperparameters from strings
    if args.hyperparameters:
        for model_type,options in args.hyperparameters.items():
            assert isinstance(options, dict)
            for k,v in options.items():
                args.hyperparameters[model_type][k] = eval(v)
        print(f'AutoGluon Hyperparameters: {args.hyperparameters}', end='\n\n')
    
    train(args)

    # Package inference code with model export
    subprocess.call('mkdir /opt/ml/model/code'.split())
    subprocess.call('cp /opt/ml/code/inference.py /opt/ml/model/code/'.split())
    
    elapsed_time = round(timer()-start,3)
    print(f'Elapsed time: {elapsed_time} seconds')  
    print('===== Training Completed =====')

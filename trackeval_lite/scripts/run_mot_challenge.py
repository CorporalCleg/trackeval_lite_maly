import sys
import os
import argparse
import pandas as pd

# Add parent directory to path to allow importing trackeval when run as a script
if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import trackeval_lite

def run_mot_challenge_evaluation(gt_path, tracker_path, seq_length, metrics=['HOTA', 'CLEAR', 'Identity'], threshold=0.5, bev=False):
    """
    Function to run MOT challenge evaluation.
    
    Args:
        gt_path (str): Path to ground truth file.
        tracker_path (str): Path to tracker results file.
        seq_length (int): Sequence length.
        metrics (list): List of metrics to evaluate.
        threshold (float): Similarity threshold.
        
    Returns:
        list: Evaluation results.
    """
    config = {
        'GT_PATH': gt_path,
        'TRACKER_PATH': tracker_path,
        'SEQ_LENGTH': seq_length,
        'METRICS': metrics,
        'THRESHOLD': threshold,
        'BEV': bev,
    }

    eval_config = {'PRINT_CONFIG': False, 'DISPLAY_LESS_PROGRESS': True}
    dataset_config = {
        'GT_PATH': gt_path,
        'TRACKER_PATH': tracker_path,
        'SEQ_LENGTH': seq_length,
        'BEV': bev,
    }
    metrics_config = {
        'METRICS': metrics,
        'THRESHOLD': threshold
    }

    evaluator = trackeval_lite.Evaluator(eval_config)
    dataset_list = [trackeval_lite.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    
    # Map metric names to classes
    metric_map = {
        'HOTA': trackeval_lite.metrics.HOTA,
        'CLEAR': trackeval_lite.metrics.CLEAR,
        'Identity': trackeval_lite.metrics.Identity,
        'VACE': trackeval_lite.metrics.VACE
    }
    
    for metric_name in metrics:
        if metric_name in metric_map:
            metric_class = metric_map[metric_name]
            metrics_list.append(metric_class(metrics_config))
            
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
        
    return evaluator.evaluate(dataset_list, metrics_list)

if __name__ == '__main__':
    # Command line interface:
    default_dataset_config = {
        'GT_PATH': None,
        'TRACKER_PATH': None,
        'SEQ_LENGTH': None,
    }
    default_metrics_config = {
        'METRICS': ['HOTA', 'CLEAR', 'Identity', 'VACE'],
        'THRESHOLD': 0.5,
    }

    config = {**default_dataset_config, **default_metrics_config}  # Merge default configs
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if setting == 'SEQ_LENGTH':
            parser.add_argument(f"--{setting}", type=int)
        elif setting in ['GT_PATH', 'TRACKER_PATH']:
            parser.add_argument(f"--{setting}", type=str)
        elif isinstance(config[setting], list) or config[setting] is None:
            parser.add_argument(f"--{setting}", nargs='+')
        else:
            parser.add_argument(f"--{setting}", type=float)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            config[setting] = args[setting]

    run_mot_challenge_evaluation(
        gt_path=config['GT_PATH'],
        tracker_path=config['TRACKER_PATH'],
        seq_length=config['SEQ_LENGTH'],
        metrics=config['METRICS'],
        threshold=config['THRESHOLD']
    )

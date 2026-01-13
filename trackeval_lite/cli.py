import sys
import os
import argparse
import trackeval_lite

def main():
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

    gt_path = config['GT_PATH']
    tracker_path = config['TRACKER_PATH']
    seq_length = config['SEQ_LENGTH']
    metrics = config['METRICS']
    threshold = config['THRESHOLD']

    # Run code
    eval_config = {'PRINT_CONFIG': False}  # Ensure PRINT_CONFIG is in eval_config
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    metrics_config['THRESHOLD'] = threshold  # Apply threshold to all metrics

    evaluator = trackeval_lite.Evaluator(eval_config)
    dataset_list = [trackeval_lite.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval_lite.metrics.HOTA, trackeval_lite.metrics.CLEAR, trackeval_lite.metrics.Identity, trackeval_lite.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)

if __name__ == '__main__':
    main()

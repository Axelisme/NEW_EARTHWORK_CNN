import time
import yaml
import argparse
import importlib
from typing import Dict, Any
from argparse import Namespace
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from util.io import show
from util.utility import init
from modules.runner import Runner
from modules.ckptmanager import CheckPointManager

def get_models(arch_conf:Dict[str, Dict|Any]):
    # select model
    model_select = arch_conf['model']['select']
    show(f"[INFO] Using {model_select} as model.")
    model_module = importlib.import_module(arch_conf['model'][model_select]['module'])
    model = getattr(model_module, model_select)(**arch_conf['model'][model_select]['args'])

    # select evaluator
    metric_selects = arch_conf['metric']['select']
    metrics = {}
    for name in metric_selects:
        show(f"[INFO] Using {name} as metric.")
        # use importlib to avoid weird bug of 'BinnedAveragePrecision' not found
        metric_module = importlib.import_module(arch_conf['metric'][name]['module'])
        metrics[name] = getattr(metric_module, name)(**arch_conf['metric'][name]['args'])
    metrics = MetricCollection(metrics)

    return model, metrics


def get_dataloader(data_conf:Dict[str, Dict|Any]):
    def get_dataset(dataset_conf, mode):
        dataset_select = dataset_conf['select']
        show(f"[INFO] Using {dataset_select} as {mode} dataset.")
        dataset_module = importlib.import_module(dataset_conf[dataset_select]['module'])
        return getattr(dataset_module, dataset_select)(**dataset_conf[dataset_select]['args'])

    valid_set = get_dataset(data_conf['valid_loader']['dataset'], 'valid')
    valid_loader = DataLoader(dataset=valid_set, **data_conf['valid_loader']['args'])

    return valid_loader


def start_train(args: Namespace, conf: Dict[str, Dict|Any]):
    """Training model base on given config."""
    # load models
    arch_conf = conf['architectures']
    model, metrics = get_models(arch_conf)

    # load model and optimizer from checkpoint if needed
    ckpt_conf = conf['ckpt']
    save_conf = {'args':vars(args), 'config':conf}
    ckpt_manager = CheckPointManager(model, save_conf, **ckpt_conf)
    ckpt_manager.save_config(f"evaluate_{time.strftime('%Y%m%d_%H%M%S')}.yaml")
    show(f"[INFO] Loading checkpoint from {args.load}")
    ckpt_manager.load(ckpt_path=args.load)

    # prepare dataset and dataloader
    data_conf = conf['data']
    valid_loader = get_dataloader(data_conf)

    # create trainer and valider
    runner_conf = conf['runner']
    runner = Runner(model=model,
                    train_loader=None,
                    valid_loader=valid_loader,
                    optimizer=None,
                    scheduler=None,
                    criterion=None,
                    metrics=metrics,
                    **runner_conf)

    # start evaluation
    runner.valid_one_epoch()
    result = runner.pop_valid_result()
    show("Evaluation result:")
    for name, evaluator in result.items():
        show(f'\t{name}: {evaluator:0.4f}')


def main():
    """Main function."""
    # parse arguments
    parser = argparse.ArgumentParser(description='Evaluate model.')
    parser.add_argument('-c', '--config', type=str, default='configs/template.yaml', help='path to config file')
    parser.add_argument('--load', required=True, type=str, help='path to ckpt to load')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    args = parser.parse_args()

    # load config
    with open(args.config, 'r') as f:
        conf = yaml.load(f, Loader=yaml.Loader)
        if 'args' in conf and 'config' in conf:
            conf = conf['config']

    # initialize
    init(0)

    # start training
    start_train(args, conf)


if __name__ == '__main__':
    main()

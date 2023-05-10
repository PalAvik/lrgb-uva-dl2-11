import os
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             )

import argparse
import torch

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained

from torch_geometric.graphgym.model_builder import create_model


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """

    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classification model')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file path',
                        required=True, type=str)
    parser.add_argument('--model_name', dest='model_name', help='name to save model under',
                        required=True, type=str)
    parser.add_argument('--device', dest='device', help='device to load model onto',
                        required=False, type=str, default='cpu')

    # Does nothing, but required for their thing
    parser.add_argument('opts', help='See graphgym/config.py for all options',
                        default=None, nargs=argparse.REMAINDER)

    return parser.parse_args()


if __name__ == '__main__':
    print("Now running!")

    # Load cmd line args
    args = parse_args()

    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)

    assert cfg.train.finetune, "Need to specify finetune==True in config file"

    cfg = load_pretrained_model_cfg(cfg)
    # Set machine learning pipeline
    # loaders = create_loader()
    # loggers = create_logger()

    print("Creating Model")
    model = create_model()
    model = init_model_from_pretrained(model, cfg.train.finetune,
                                       cfg.train.freeze_pretrained,
                                       device=args.device)

    print('Model created and weights loaded')

    print('Saving serialised model!')

    out_path = os.path.join(f'./entire_models/{args.model_name}')

    with open(out_path, 'w+') as f:
        torch.save(model, f)

    print(f'Saved to {out_path}')

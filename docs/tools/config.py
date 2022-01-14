from argparse import Namespace
import logging
import sys
from datetime import datetime
from os import path
from random import randint

import toml
import torch
import torch.backends.cudnn as cudnn

from utils import misc


def write(bunch, path):
    with open(path, 'w') as fp:
        toml.dump(bunch, fp)


def load_args(path, kw):
    opt = toml.load(path)
    opt.update(kw)
    bunch = Namespace(**opt)
    return bunch


def init(opt, toml_path, *, save_name='', seed=-1):
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _opt = {
        'seed': randint(0, 12345) if seed == -1 else seed,
        'save': save_name if save_name else time_stamp,
        'save_path': f"./results{save_name}"
    }
    opt.update(_opt)
    args = load_args(toml_path, opt)

    torch.manual_seed(args.seed)
    # cuda
    if 'cuda' in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device_ids[0])
        cudnn.benchmark = True
    else:
        args.device_ids = None

    # set logs
    misc.mkdir(args.save_path)
    misc.setup_logging(path.join(args.save_path, 'log.txt'))

    # print logs
    logging.info(args)
    return args

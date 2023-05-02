#!/usr/bin/python
# -*- coding: utf-8 -*-
import time, pdb, argparse, subprocess
from pathlib import Path

import torch.multiprocessing

from SyncNetInstance import *


# ==================== LOAD PARAMS ====================

def get_parser():
    parser = argparse.ArgumentParser(description="SyncNet")

    parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='')
    parser.add_argument('--batch_size', type=int, default='20', help='')
    parser.add_argument('--vshift', type=int, default='15', help='')
    parser.add_argument('--data-path', type=str, default="data", help='')
    parser.add_argument('--tmp_dir', type=str, default="data/work/pytmp", help='')
    parser.add_argument('--reference', type=str, default="demo", help='')

    opt = parser.parse_args()
    return opt


# ==================== RUN EVALUATION ====================

def run_eval(opt, device):
    s = SyncNetInstance()
    s = s.to(device)

    s.loadParameters(opt.initial_model)
    # print("Model %s loaded."%opt.initial_model);

    conf, dist = s.evaluate(opt, videofile=opt.videofile)

    return dict(vid_name=opt.videofile, confidence=conf, distance=dist)


if __name__ == '__main__':
    args = get_parser()
    torch.multiprocessing.set_start_method("spawn")
    p = Path(args.data_path)
    vid_files = list(p.rglob("*.mp4"))
    num_gpus = torch.cuda.device_count()
    commands = [(args, f, i % num_gpus) for i, f in enumerate(vid_files)]

    results = dict()
    with torch.multiprocessing.Pool(torch.cuda.device_count()) as pool:
        results.update(pool.starmap(run_eval, commands))

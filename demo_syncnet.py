#!/usr/bin/python
# -*- coding: utf-8 -*-
import pickle
import time, pdb, argparse, subprocess
from collections import defaultdict
from pathlib import Path

import torch.multiprocessing

from SyncNetInstance import *


# ==================== LOAD PARAMS ====================


def get_parser():
    parser = argparse.ArgumentParser(description="SyncNet")

    parser.add_argument(
        "--initial_model", type=str, default="data/syncnet_v2.model", help=""
    )
    parser.add_argument("--batch_size", type=int, default="20", help="")
    parser.add_argument("--vshift", type=int, default="15", help="")
    parser.add_argument("--data-path", type=str, default="data", help="")
    parser.add_argument("--tmp_dir", type=str, default="data/work/pytmp", help="")
    parser.add_argument("--reference", type=str, default="demo", help="")

    opt = parser.parse_args()
    return opt


# ==================== RUN EVALUATION ====================


def run_eval(opt, filename, device=0):
    s = SyncNetInstance()
    s = s.to(device)

    s.loadParameters(opt.initial_model)
    # print("Model %s loaded."%opt.initial_model);

    conf, dist = s.evaluate(opt, videofile=filename)

    return f"{filename.parent.stem}_{filename.stem}", conf, dist


if __name__ == "__main__":
    args = get_parser()
    p = Path(args.data_path)
    vid_files = p.rglob("*.mp4")

    results = defaultdict(list)
    de_vids = 0
    for f in vid_files:
        try:
            r = run_eval(args, f)
        except:
            de_vids += 1
            r = ["defect_video", sum(results["confidence"]) / len(results["confidence"]), sum(results["distance"]) / len(results["distance"])]
        results["filename"].append(r[0])
        results["confidence"].append(r[1])
        results["distance"].append(r[2])

    with open(f"data/{p.stem}_results.pkl", "wb") as file:
        pickle.dump(results, file)
    print(10 * "#", "\n", f"NUMBER OF DAMAGED VIDS: {de_vids}\n", 10 * "#")

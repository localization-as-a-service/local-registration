import open3d
import numpy as np
import pandas as pd
import os
import tqdm
import copy
import glob

from scipy.signal import argrelmin
from PIL import Image
from utils.config import Config

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import utils.registration as registration
import utils.fread as fread
import utils.FCGF as FCGF
import utils.helpers as helpers

plt.rcParams.update({'font.size': 8})


def find_cutoffs(std_values, target_fps, min_std):
    cutoffs = argrelmin(std_values, order=target_fps // 2)[0]
    return cutoffs[np.where(np.abs(std_values[cutoffs] - min_std) < 0.5)[0]]


def plot_cutoffs(config: Config, show=False, save=True):
    sequence_dir = config.get_sequence_dir(include_secondary=True)
    feature_dir = config.get_feature_dir()
    
    sequence_ts = fread.get_timstamps(feature_dir, ext=".secondary.npz")
    num_frames = len(sequence_ts)

    print("     :: Number of frames: {}".format(num_frames))
    
    std_values = []

    for t in tqdm.trange(len(sequence_ts)):
        depth_img = Image.open(os.path.join(sequence_dir, f"frame-{sequence_ts[t]}.depth.png")).convert("I")
        depth_img = np.array(depth_img) / 4000
        std_values.append(np.std(depth_img))
        
    std_values = np.array(std_values)
    
    cutoffs = find_cutoffs(std_values, config.target_fps, config.min_std)
    min_indices = argrelmin(std_values, order=config.target_fps // 2)[0]
    
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, len(min_indices))

    ax = fig.add_subplot(gs[0, :])

    ax.plot(std_values)
    ax.scatter(min_indices, std_values[min_indices], c="b", marker="x")

    for x in cutoffs:
        ax.axvline(x - config.cutoff_margin, c="g", linestyle="--")
        ax.axvline(x, c="b", linestyle="--")
        ax.axvline(x + config.cutoff_margin, c="g", linestyle="--")
        
    ax.axhline(y=config.min_std, color="r", linestyle="--")
    ax.set_ylim(0, 4)
    ax.set_xlim(0, len(std_values))

    ax.set_ylabel("Frame #")
    ax.set_xlabel("Std. of Distances to the camera")

    for i in range(len(min_indices)):
        ax = fig.add_subplot(gs[1, i])
        img = Image.open(os.path.join(sequence_dir, f"frame-{sequence_ts[min_indices[i]]}.depth.png")).convert("I").convert("I")
        ax.imshow(np.asarray(img), cmap="jet")
        ax.axis("off")
        ax.set_title(f"#{min_indices[i]}")

    plt.tight_layout()
    if show:
        plt.show()

    if save:
        plt.savefig(config.get_output_file(filename=f"{config.get_file_name()}.png"))
    
    plt.close()

if __name__ == "__main__":
    sequence = 1
    config = Config(
        feature_dir="data/features",
        sequence_dir="data/raw_data",
        experiment="exp_10",
        trial="trial_1",
        subject="subject-1",
        sequence=f"{sequence:02d}",
        groundtruth_dir="data/trajectories/groundtruth",
        output_dir="results/cutoffs",
        min_std=0.5,
        cutoff_margin=5,
        target_fps=20
    )
    
    for trial in os.listdir(os.path.join(config.feature_dir, config.experiment)):
        config.trial = trial
        for subject in os.listdir(os.path.join(config.feature_dir, config.experiment, config.trial, str(config.voxel_size))):
            config.subject = subject    
            for sequence in os.listdir(os.path.join(config.feature_dir, config.experiment, config.trial, str(config.voxel_size), config.subject)):
                config.sequence = sequence
                print(f"Processing: {config.experiment} >> {config.trial} >> {config.subject} >> {config.sequence}")
                plot_cutoffs(config, save=True, show=False)
                
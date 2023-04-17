import os
import cv2
import utils.fread as fread
from utils.config import Config
import tqdm
import numpy as np

def make_video(config: Config, depth: bool = False):
    seqeunce_ts = fread.get_timstamps_from_images(config.get_sequence_dir(), ext="color.png")
    images = []
    
    extension = "depth" if depth else "color"

    for t in seqeunce_ts:
        if not depth:
            img = cv2.imread(os.path.join(config.get_sequence_dir(), f"frame-{t}.color.png"))
        else:
            img = cv2.imread(os.path.join(config.get_sequence_dir(), f"frame-{t}.depth.png"), cv2.IMREAD_ANYDEPTH)
            img = img / (9 * 4000)
            img = img * 255
            img = np.where(img > 255, 255, img)
            img = np.array(img, dtype=np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        images.append(img)

    video = cv2.VideoWriter(f"{config.get_output_file(config.get_file_name())}.{extension}.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
    for image in tqdm.tqdm(images):
        video.write(image)
    
    
if __name__ == "__main__":
    config = Config(
        sequence_dir="data/raw_data",
        feature_dir="data/features",
        output_dir="data/videos",
        experiment="exp_8",
        trial="trial_3",
        subject="subject-1",
        sequence="01",
        groundtruth_dir="data/trajectories/groundtruth",
    )
    
    config.voxel_size=0.03
    config.target_fps=20
    config.min_std=0.5
    
    for trial in os.listdir(os.path.join(config.feature_dir, config.experiment)):
        config.trial = trial
        if trial == "trial_4" or trial == "trial_1":
            continue
        for subject in os.listdir(os.path.join(config.feature_dir, config.experiment, config.trial, str(config.voxel_size))):
            config.subject = subject    
            for sequence in os.listdir(os.path.join(config.feature_dir, config.experiment, config.trial, str(config.voxel_size), config.subject)):
                config.sequence = sequence
                print(f"Processing: {config.experiment} >> {config.trial} >> {config.subject} >> {config.sequence}")
                make_video(config, depth=False)
                
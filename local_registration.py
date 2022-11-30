import open3d
import numpy
import os
import utils.fread as fread
import utils.FCGF as FCGF
import utils.registration as registration


def main():
    sequence_dir = "../localization-data/data/features/exp_5/trial_1/0.03/subject-1/01"
    sequence_ts = fread.get_timstamps(sequence_dir)
    
    source = FCGF.get_features(os.path.join(sequence_dir, f"{sequence_ts[0]}.secondary.npz"), pcd_only=True)
    
    
if __name__ == "__main__":
    main()

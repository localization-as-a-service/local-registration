import os
from dataclasses import dataclass

@dataclass
class Config:
    feature_dir: str
    experiment: str
    trial: str
    subject: str
    sequence: str
    sequence_dir: str
    output_dir: str
    groundtruth_dir: str
    
    voxel_size: float = 0.03
    target_fps: int = 20
    
    min_std: float = 0.6
    threshold: float = 0.5
    cutoff_margin: int = 5
    delta: float = 0.1
    
    def get_sequence_dir(self, include_secondary: bool = False):
        if include_secondary:
            return os.path.join(self.sequence_dir, self.experiment, self.trial, "secondary", self.subject, self.sequence, "frames")
        return os.path.join(self.sequence_dir, self.experiment, self.trial, self.subject, self.sequence, "frames")
    
    def get_motion_dir(self, include_secondary: bool = False):
        if include_secondary:
            return os.path.join(self.sequence_dir, self.experiment, self.trial, "secondary", self.subject, self.sequence, "motion")
        return os.path.join(self.sequence_dir, self.experiment, self.trial, self.subject, self.sequence, "motion")
    
    def get_feature_dir(self):
        return os.path.join(self.feature_dir, self.experiment, self.trial, str(self.voxel_size), self.subject, self.sequence)
    
    def get_groundtruth_dir(self):
        return os.path.join(self.groundtruth_dir, self.experiment)
    
    def get_file_name(self):
        return f"{self.experiment}__{self.trial}__{self.subject}__{self.sequence}"
    
    def get_output_file(self, filename: str):
        if not os.path.exists(os.path.join(self.output_dir, self.experiment)): 
            os.makedirs(os.path.join(self.output_dir, self.experiment))
        
        return os.path.join(self.output_dir, self.experiment, filename)
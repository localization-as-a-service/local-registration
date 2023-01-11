import os
import cv2
import utils.fread as fread
import argparse
import tqdm

parser = argparse.ArgumentParser(description='Convert sequence to video')
parser.add_argument('--dir', type=str, help='Path to main directory')
parser.add_argument('--exp', type=int, help='Experiment number')
parser.add_argument('--trial', type=int, help='Trial number')
parser.add_argument('--subject', type=int, help='Subject number')
parser.add_argument('--seq', type=int, help='Sequence number')
parser.add_argument('--output', type=str, help='Output directory')

args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)

sequence_dir = os.path.join(args.dir, f"exp_{args.exp}", f"trial_{args.trial}", f"subject-{args.subject}", f"{args.seq:02d}", "frames")

ts = fread.get_timstamps_from_images(sequence_dir, ext="color.png")

images = []

for t in ts:
    if (t - ts[0]) / 1e9 > 11: break 
    img = cv2.imread(os.path.join(sequence_dir, f"frame-{t}.color.png"))
    images.append(img)


output_file = os.path.join(args.output, f"exp_{args.exp}_trial_{args.trial}_subject_{args.subject}_{args.seq:02d}.avi")
video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))

for image in tqdm.tqdm(images):
    video.write(image)
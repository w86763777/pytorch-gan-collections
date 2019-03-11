import argparse
import os

import imageio
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--log-dir', type=str, default='log')
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=1000000000)
parser.add_argument('--frame', type=int, default=50)
args = parser.parse_args()

sample_dir = os.path.join(args.log_dir, args.name, 'sample')
file_names = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
file_names = sorted(file_names, key=lambda f: int(f.split('.')[0]))
file_names = filter(lambda f: int(f.split('.')[0]) <= args.end, file_names)
file_names = list(file_names)
file_names = file_names[args.start:args.end:len(file_names) // args.frame]
file_names = file_names[:args.frame]

frames = []
for file_name in tqdm(file_names, desc='Load image'):
    frames.append(imageio.imread(os.path.join(sample_dir, file_name)))
print('Save gif...')
imageio.mimsave("%s.gif" % args.name, frames, 'GIF', duration=0.01)

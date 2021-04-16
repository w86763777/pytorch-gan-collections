import os
import glob

import imageio
import numpy as np
from absl import app, flags


FLAGS = flags.FLAGS
flags.DEFINE_string('sample_dir', None, 'path to logdir')
flags.DEFINE_string('output', None, 'path to logdir')
flags.DEFINE_integer('frame_num', 50, 'number of frames')
flags.mark_flag_as_required('sample_dir')
flags.mark_flag_as_required('output')


def main(argv):
    file_names = glob.glob(os.path.join(FLAGS.sample_dir, '*.png'))
    file_names = sorted(
        file_names,
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    file_names = file_names[::len(file_names) // FLAGS.frame_num]

    frames = []
    for file_name in file_names:
        frames.append(imageio.imread(file_name))
    druation = list(np.linspace(0.05, 0.2, len(file_names)))
    print('Save gif...')
    imageio.mimsave(FLAGS.output, frames, 'GIF', duration=druation)


if __name__ == '__main__':
    app.run(main)

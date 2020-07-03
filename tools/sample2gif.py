import os

import imageio
from absl import app, flags
from tqdm import tqdm


FLAGS = flags.FLAGS
flags.DEFINE_string('logdir', None, 'path to logdir')
flags.DEFINE_integer('frame_num', 50, 'number of frames')
flags.mark_flag_as_required('logdir')


def main(argv):
    sample_dir = os.path.join(FLAGS.logdir, 'sample')
    file_names = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
    file_names = sorted(file_names, key=lambda f: int(f.split('.')[0]))
    file_names = file_names[::len(file_names) // FLAGS.frame_num]

    frames = []
    for file_name in tqdm(file_names, desc='Load image'):
        frames.append(imageio.imread(os.path.join(sample_dir, file_name)))
    print('Save gif...')
    imageio.mimsave(
        os.path.join(FLAGS.logdir, 'progress.gif'), frames, 'GIF',
        duration=0.01)


if __name__ == '__main__':
    app.run(main)

import numpy as np
import torch
from tqdm import trange

from .inception import InceptionV3
from .fid_score import calculate_frechet_distance


def get_inception_and_fid_score(images, device, fid_cache, is_splits=10,
                                batch_size=50, verbose=False):
    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    block_idx2 = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
    model = InceptionV3([block_idx1, block_idx2]).to(device)
    model.eval()

    if batch_size > len(images):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(images)

    fid_acts = np.empty((len(images), 2048))
    is_probs = np.empty((len(images), 1008))

    if verbose:
        iterator = trange(0, len(images), batch_size)
    else:
        iterator = range(0, len(images), batch_size)

    for start in iterator:
        end = start + batch_size
        batch_images = images[start: end]

        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = model(batch_images)
        fid_acts[start: end] = pred[0].view(-1, 2048).cpu().numpy()
        is_probs[start: end] = pred[1].cpu().numpy()

    # Inception Score
    scores = []
    for i in range(is_splits):
        part = is_probs[
            (i * is_probs.shape[0] // is_splits):
            ((i + 1) * is_probs.shape[0] // is_splits), :]
        kl = part * (
            np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    is_score = (np.mean(scores), np.std(scores))

    # FID Score
    m1 = np.mean(fid_acts, axis=0)
    s1 = np.cov(fid_acts, rowvar=False)
    f = np.load(fid_cache)
    m2, s2 = f['mu'][:], f['sigma'][:]
    f.close()
    fid_score = calculate_frechet_distance(m1, s1, m2, s2)

    return is_score, fid_score

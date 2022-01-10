import os
import glob

import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    """
    Manually download csv file of Inception Score and FID from tensorboard
    """
    IS = []
    FID = []
    for path in glob.glob('./logs/*.csv'):
        df = pd.read_csv(path)
        _, name, _, tag = os.path.splitext(
            os.path.basename(path))[0].split('-')
        if '_CIFAR10_' in name:
            name = name.replace('_CIFAR10_', '(')
            name = name + ')'
        if name.endswith('_CIFAR10'):
            name = name.replace('_CIFAR10', '')
        if tag == 'Inception_Score':
            IS.append((name, df.values[:, 1], df.values[:, 2]))
        elif tag == 'FID':
            FID.append((name, df.values[:, 1], df.values[:, 2]))
        else:
            raise ValueError("???")
    IS = sorted(IS, key=lambda x: x[2][-1], reverse=True)
    FID = sorted(FID, key=lambda x: x[2][-1])

    for name, x, y in IS:
        plt.plot(x, y, label=name)
    plt.legend()
    plt.xlabel('Step', fontsize=16)
    plt.ylabel('Inception Score', fontsize=16)
    # plt.title('Inception Score', fontsize=16)
    plt.savefig('./IS.png')
    plt.cla()

    for name, x, y in FID:
        plt.plot(x, y, label=name)
    plt.legend()
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))
    plt.ylabel('FID', fontsize=16)
    plt.xlabel('Step', fontsize=16)
    # plt.title('FID curve', fontsize=16)
    plt.savefig('./FID.png')

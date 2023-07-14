import numpy as np


def plot(_range: range, results: dict, title:str):
    iters = np.arange(0, len(_range), 1)
    label1, label2, label3 = results.keys()
    plt.plot(iters, results[label1], label=label1)
    plt.plot(iters, results[label2], label=label2)
    plt.plot(iters, results[label3], label=label3)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title(title)
    plt.legend()
    plt.savefig(f'{title}.png')

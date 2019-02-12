import matplotlib.pyplot as plt
import numpy as np

def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    for i, key in enumerate(hist.keys()):
        plt.plot(range(len(hist[key])), hist[key], label=key)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()

def save_histogram(inputs, bins=np.linspace(0,2, 1000), path='histogram.png'):
    for index, input in enumerate(inputs):
        input = np.array(input)
        if index == 0:
            max_value = np.max(input)
            min_value = np.min(input)
            mean_value = np.mean(input)
        else :
            max_value = np.max(input) if max_value<np.max(input) else max_value
            mean_value = np.mean(input) if mean_value<np.mean(input) else mean_value
            min_value = np.min(input) if min_value>np.min(input) else min_value
    bins = np.linspace(min_value, max_value, 10000)
    for index, input in enumerate(inputs) :
        # print("11111")
        # print(np.array(list(np.array(input).flatten())).shape)
        y,x,_ = plt.hist(np.array(input).flatten(), alpha=0.5, bins=1000, label=str(index))
        print(y.max())
    plt.legend(loc='upper right')
    plt.yscale('log', nonposy='clip')
    plt.savefig(path)
    plt.close()






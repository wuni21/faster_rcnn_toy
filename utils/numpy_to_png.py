import matplotlib.pyplot as plt
import itertools
import numpy as np
import torch

def save_as_group(input, name, num_epoch, show=False, save=False, path='result.png', isFix=False):
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for k in range(5*5):
        i=k // 5
        j=k % 5
        ax[i, j].cla()
        if input.size()[1] == 1:
            # print(torch.sum(input>1))
            # print(torch.sum(input<0))
            # print(input[k, :,:,:].cpu().data.view(28,28).dtype)
            # print(input[k, :,:,:].cpu().data.view(28,28))
            ax[i, j].imshow(input[k, :,:,:].cpu().data.view(input.size()[2],input.size()[3]).numpy(), cmap='gray')
        elif input.size()[1] == 3 :
            image = np.array(input[k, :, :, :].cpu().data.view(3, 32, 32), np.float32)
            # image = (image-image.min())/(image.max()-image.min())
            ax[i, j].imshow(image.transpose(1,2,0))


    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    path = path + name
    plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()




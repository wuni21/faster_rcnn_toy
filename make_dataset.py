import scipy.io as sio
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


data = sio.loadmat('mnist_data.mat')
# data2 = sio.loadmat('detecti_mnist.mat')


train_data = np.array(data['train_28'])
train_label = np.array(data['label_train'])
# print(train_label[0])
train_label = np.argmax(train_label, axis=1)
# print(train_label)
total_num = train_data.shape[0]
# print(total_num)
# train_data = 255-train_data
# print(train_data)

# im_info = [np.array([400,400,1])]*100
gt_boxes = []
labels = []
num_boxes = []



train_data = np.squeeze(train_data)
scale_matrix = [4]

im_data = np.array([])

for i in range(400):
    print(i)
    num = np.random.randint(2)+2
    noise = np.random.normal(0,0.2,(100,100))
    
    noisy_background = np.clip(noise,0,1) * 255
    #print(noisy_background)
    background = noisy_background
    tmp = np.zeros([100,100])
    tmp_box = []
    tmp_label = []
    loc = np.random.permutation(9)

    for j in range(2):
        count = num
        index = np.random.randint(total_num)
        selected = train_data[index]
        selected_label = train_label[index]
        # x_start = np.random.randint(72)
        # y_start = np.random.randint(72)
        x_start = (loc[j] - loc[j]//3 * 3) * 28 + 10
        y_start = loc[j]//3 * 28 + 10
        # x_scale = scale_matrix[np.random.randint(3)]
        # y_scale = scale_matrix[np.random.randint(3)]
        x_scale = 1
        y_scale = 1
        x_size = int(x_scale*28)
        y_size = int(y_scale * 28)
        x_end = x_start + x_size
        y_end = y_start + y_size
        if x_end>100 or y_end>100 :
            tmp = tmp
            count = count-1
        else:
            # print(scipy.misc.imresize(selected, (y_size, x_size)).shape)
            # print(tmp[y_start:y_end, x_start:x_end].shape)
            tmp[y_start:y_end, x_start:x_end] = scipy.misc.imresize(selected, (y_size, x_size))
            # print(tmp_box)
            # print(np.array([y_start, x_start, y_end, x_end, selected_label+1]))
            tmp_box.append(np.array([y_start, x_start, y_end, x_end]))
            tmp_label.append(np.array([selected_label+1]))
            # print(tmp_box)
    result = np.clip(background+tmp, 0, 255)
    #plt.imshow(result.squeeze(), cmap='gray')
    #plt.savefig('generated.png')
    gt_box = np.zeros([2, 4])
    label = np.zeros([2])
    # print(np.expand_dims(np.stack(tmp_box, axis=0), axis=0).shape)
    gt_box[0:count, :] = tmp_box
    label[0:count] = tmp_label
    if im_data.size == 0:
        im_data = np.expand_dims(np.expand_dims(result, 2), 0)
        # print(im_data.shape)
    else:
        im_data = np.concatenate((im_data, np.expand_dims(np.expand_dims(result, 2), 0)), axis=0)

    # num_boxes.append(count)
    # gt_boxes.append(np.array(gt_box))
        # print(im_data.shape)

    gt_boxes.append(gt_box)
    labels.append(label)
    # print(labels)
    # print("sadfsadfasd")

    # print(gt_boxes)
    # print(labels)
# print(im_data.shape)

# im_data = np.squeeze(im_data)
print(im_data.shape)
print(im_data.dtype)
im_data = im_data.astype(np.uint8)
gt_boxes = np.stack(gt_boxes)
print(gt_boxes.shape)
labels = np.stack(labels)
print(labels.shape)

sio.savemat('detect_mnist_small.mat', {'im_data':im_data, 'gt_boxes':gt_boxes, 'labels':labels})
plt.imshow(im_data[0].squeeze(), cmap='gray')
plt.savefig('generated.png')


fig, ax = plt.subplots()
ax.imshow(np.squeeze(im_data[0])/255, cmap='gray')
rect = patches.Rectangle(xy=(100, 100), width=100, height=100, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.savefig('im_gt_loc')


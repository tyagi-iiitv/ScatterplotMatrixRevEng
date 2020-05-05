from os import listdir
from os import path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from PIL import ImageOps
import numpy as np
import cv2
from skimage.transform import rescale, resize
from natsort import natsorted

def remove_fg(res2, patch_img):
    zero_list = list(res2[0,:,0]) + list(res2[:,0,0]) + list(res2[-1,:,0]) + list(res2[:,-1,0])
    one_list = list(res2[0,:,1]) + list(res2[:,0,1]) + list(res2[-1,:,1]) + list(res2[:,-1,1])
    two_list = list(res2[0,:,2]) + list(res2[:,0,2]) + list(res2[-1,:,2]) + list(res2[:,-1,2])
    zero_pix = max(set(zero_list), key=zero_list.count) # Background pixels
    one_pix = max(set(one_list), key=one_list.count)
    two_pix = max(set(two_list), key=two_list.count)
    zero_min = list(np.unique(res2[:,:,0]))
    one_min = list(np.unique(res2[:,:,1]))
    two_min = list(np.unique(res2[:,:,2]))
    zero_min.remove(zero_pix) # Foreground pixels
    one_min.remove(one_pix)
    two_min.remove(two_pix)
    # Creating a mask
    mask = res2[:,:,0]
    mask = np.where(mask==zero_min, 0, 1)
    # Expand mask by some pixels
    exp_factor = 3
    just_changed = np.zeros(mask.shape, dtype=bool)
    for i in range(patch_img.shape[0]-exp_factor):
        for j in range(patch_img.shape[1]-exp_factor):
            if mask[i][j] == 0:
                continue
            found = False
            for k in range(1,exp_factor):
                if (
                    (mask[i+k][j] == 0 and not just_changed[i+k][j]) or 
                    (mask[i-k][j] == 0 and not just_changed[i-k][j]) or
                    (mask[i][j+k] == 0 and not just_changed[i][j+k]) or
                    (mask[i][j-k] == 0 and not just_changed[i][j-k])
                    ):
                    found = True
                    break
            if found:
                mask[i][j] = 0
                just_changed[i][j] = True

    patch_img2 = np.array(patch_img)
    img_bg = patch_img2
    mask = mask.astype(np.uint8)
    img_bg[:,:,0] = cv2.multiply(mask, patch_img2[:,:,0])
    img_bg[:,:,1] = cv2.multiply(mask, patch_img2[:,:,1])
    img_bg[:,:,2] = cv2.multiply(mask, patch_img2[:,:,2])
    img_fg = np.array(patch_img)
    img_fg[:,:,0] = zero_pix
    img_fg[:,:,1] = one_pix
    img_fg[:,:,2] = two_pix
    img_fg[:,:,0] = cv2.multiply(1-mask, img_fg[:,:,0])
    img_fg[:,:,1] = cv2.multiply(1-mask, img_fg[:,:,1])
    img_fg[:,:,2] = cv2.multiply(1-mask, img_fg[:,:,2])
    out_image = cv2.add(img_fg, img_bg)
    return out_image



train_imgs = natsorted(listdir("./images/train"))
print("train_imgs")
bg_train = train_imgs
test_imgs = natsorted(listdir("./images/test"))
print("test_imgs")
bg_test = test_imgs
train_labels = natsorted(listdir("./labels/train"))
print("train_labels")
test_labels = natsorted(listdir("./labels/test"))
print("test_labels")

train_imgs = ["./images/train/"+s for s in train_imgs]
test_imgs = ["./images/test/"+s for s in test_imgs]
train_labels = ["./labels/train/"+s for s in train_labels]
test_labels = ["./labels/test/"+s for s in test_labels]
bg_train = ["./background/train/"+s for s in bg_train]
bg_test = ["./background/test/"+s for s in bg_test]

images = train_imgs + test_imgs
bg_images = bg_train + bg_test
labels = train_labels + test_labels

for i, image in enumerate(images):
    label = labels[i]
    bg_image = bg_images[i]
    print(bg_image)
    if path.exists(bg_image):
        continue
    bb = pd.read_csv(label, delimiter=' ', header=None)
    data = np.array(plt.imread(image))
    h = data.shape[0]
    w = data.shape[1]
    resize_factor = 0
    for row in bb.values[:,1:]:
        y1 = row[1]*h - (row[3]*h)/2
        y2 = row[1]*h + (row[3]*h)/2
        x1 = row[0]*w - (row[2]*w)/2
        x2 = row[0]*w + (row[2]*w)/2
        width, height = x2-x1, y2-y1
        patch_img = data[int(y1):int(y2), int(x1):int(x2)]
        img = patch_img
        try:
            img_lg = resize(img, (img.shape[0]+resize_factor, img.shape[1]+resize_factor), preserve_range=True)
        except:
            continue
        Z = img_lg.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img_lg.shape))
        try:
            img_no_fg = remove_fg(res2, patch_img)
            data[int(y1):int(y2), int(x1):int(x2)] = img_no_fg
        except:
            continue
    plt.imsave(bg_image, data)
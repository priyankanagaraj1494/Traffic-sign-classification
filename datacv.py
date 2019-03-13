import tensorflow as tf
import os
import skimage
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from skimage import transform
from skimage.color import rgb2gray

config = tf.ConfigProto(log_device_placement = True, allow_soft_placement = True) #config for session
data_dir = "C:/PRIYANKA/traffic"
train_data_dir = os.path.join(data_dir, "BelgiumTSC/training")
test_data_dir = os.path.join(data_dir, "BelgiumTSC/testing")

def load_data(data_dir): #training
    dir = [d for d in os.listdir(data_dir) #000,001,0002....
           if os.path.isdir(os.path.join(data_dir,d))]
    labels = []
    images = []
    for d in dir: #looping over 0001....
        label_dir = os.path.join(data_dir, d) #62 labels
        file_names = [os.path.join(label_dir,f) #filenames.ppm
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images,labels

images,labels = load_data(train_data_dir)
images28 = [transform.resize(image,(28,28)) for image in images]
images28=np.array(images28)
print("Resized image shape",images28.shape)
images28 = rgb2gray(images28)

test_images,test_labels = load_data(test_data_dir)
test_images28 =[transform.resize(img,(28,28)) for img in test_images]
test_images28 = rgb2gray(np.array(test_images28))


"""
#view image details
images = np.array(images)
print(images.shape)
print(images.size)
labels= np.array(labels)
print(labels.ndim)
print(labels.size)
print(len(set(labels)))
#plt.hist(labels,62)
#plt.show()
"""
"""
#display traffic signs
signs = [300,250,3650,4000]
for i in range(len(signs)):
    plt.subplot(2,2,i+1) #(m,n,p) m-n grid, p is pos
    plt.axis('off')
    plt.imshow(images[signs[i]])
    plt.subplots_adjust(wspace=0.5)
    print("shape:{0}, min:{1},max:{2}".format(images[signs[i]].shape,
          images[signs[i]].min(),
          images[signs[i]].max()))
plt.show()"""

"""
#display all 62 signs
uni_labels = set(labels)
plt.figure(figsize = (18,18))
i=1
for label in uni_labels:
    img = images[labels.index(label)]
    plt.subplot(8,8,i)
    #plt.title("label{0}({1})".format(label,labels.count(label)))
    i+=1
    plt.imshow(img)
plt.show()"""


# displaying gray images
"""signs = [300,250,3650,4000]
for i in range(len(signs)):
    plt.subplot(2,2,i+1) #(m,n,p) m-n grid, p is pos
    plt.axis('off')
    plt.imshow(images28[signs[i]],cmap ="gray")
    plt.subplots_adjust(wspace=0.5)
plt.show()"""
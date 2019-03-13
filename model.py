import tensorflow as tf
from datacv import *
from tqdm import tqdm
import random


#model
tf.set_random_seed(1234)
x = tf.placeholder(dtype =tf.float32, shape =[None,28,28])
y = tf.placeholder(dtype=tf.int32,shape=[None])
#flatten (None,28,28)--(None,784)
img_flat = tf.contrib.layers.flatten(x)
#FC layer
logits = tf.contrib.layers.fully_connected(img_flat,62,tf.nn.relu)
#loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits =logits))
#optimizer
train_op = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)
#convert logits to index label , returns largest value in tensor
pred = tf.argmax(logits,1)
accuracy = tf.reduce_mean(tf.cast(pred,tf.float32)) #cast the input to new type

print("images_flat: ", img_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", pred)

sess = tf.Session()
#set train = False for evaluation
train = False
validate= False
test = True
if train:
    print("--training--")
    sess.run(tf.global_variables_initializer())
    for i in tqdm(range(2000)):
        print('epoch',i)
        _,accuracy_val,loss_val = sess.run([train_op,accuracy,loss],feed_dict ={x:images28, y:labels})
        if i%10 == 0:
            print("accuracy:",accuracy_val)
            print("loss:",loss_val)
    sess.close()

if validate:
    print("---eval---")
    sample_indexes = random.sample(range(len(images28)), 10)
    sample_images = [images28[i] for i in sample_indexes]
    sample_labels = [labels[i] for i in sample_indexes]
    sess.run(tf.global_variables_initializer())
    predicted = sess.run([pred],feed_dict={x:sample_images})[0]
    #print real and pred labels
    print("actual label",sample_labels)
    print("predicted label",predicted)
    #display
    fig = plt.figure(figsize =(10,10))
    for i in range(len(sample_images)):
        truth = sample_labels[i]
        prediction = predicted[i]
        plt.subplot(5,2,1+i)
        plt.axis('off')
        color = 'yellow' if truth == prediction else 'red'
        plt.text(30,10, "truth:{0}\n prediction:{1}".format(truth,prediction),fontsize=12)
        plt.imshow(sample_images[i],  cmap="gray")
    plt.show()
    sess.close()

if test:
    print("--testing--")
    sess.run(tf.global_variables_initializer())
    #run pred on entire test set
    predict = sess.run([pred],feed_dict={x:test_images28})[0]
    match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predict)])
    #calc accuracy
    accuracy = match_count / len(test_labels)
    print("Accuracy: {:.3f}".format(accuracy))
    sess.close()
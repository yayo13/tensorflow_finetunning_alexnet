#!/usr/bin/env python
#-*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
import os
from alexnet_model import alexnet
from caffe_classes import class_names
from datagenerator import ImageDataGenerator

class alexnet_test(object):
    def __init__(self):
        self.PRE_MODEL = 'bvlc_alexnet.npy'
    
    def test_imagenet(self, imgs_):
        num_classes = 1000
        skip_layer = []
        imgs = []
        
        #mean of imagenet dataset in BGR
        imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
        #plot images
        fig = plt.figure(figsize=(15,6))
        for i, img_ in enumerate(imgs_):
            img = cv2.imread(img_)
            imgs.append(img)
            fig.add_subplot(1,3,i+1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        
        #placeholder for input and dropout rate
        x = tf.placeholder(tf.float32, [1, 227, 227, 3])
        keep_prob = tf.placeholder(tf.float32)
        #create model with default config ( == no skip_layer and 1000 units in the last layer)
        model = alexnet(x, keep_prob, num_classes, skip_layer, weights_path=self.PRE_MODEL)
        #define activation of last layer as score
        score = model.fc8
        #create op to calculate softmax 
        softmax = tf.nn.softmax(score)

        with tf.Session() as sess:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # Load the pretrained weights into the model
            model.load_initial_weights(sess)
            # Create figure handle
            fig2 = plt.figure(figsize=(15,6))
            # Loop over all images
            for i, image in enumerate(imgs):
                # Convert image to float32 and resize to (227x227)
                img = cv2.resize(image.astype(np.float32), (227,227))
                # Subtract the ImageNet mean
                img -= imagenet_mean
                # Reshape as needed to feed into model
                img = img.reshape((1,227,227,3))
                # Run the session and calculate the class probability
                probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})
                # Get the class name of the class with the highest probability
                class_name = class_names[np.argmax(probs)]
                # Plot image with class name and prob in the title
                fig2.add_subplot(1,3,i+1)
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title("Class: " + class_name + ", probability: %.4f" %probs[0,np.argmax(probs)])
                plt.axis('off')
        plt.show()


def main():
    # test imagenet
    alex = alexnet_train()
    images = ['imgs/llama.jpeg', 'imgs/sealion.jpeg', 'imgs/zebra.jpeg']
    alex.test_imagenet(images)
    

if __name__ == '__main__':
    main()                

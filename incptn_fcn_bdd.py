#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:06:18 2018

@author: windowsx
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data
import scipy.misc
from glob import glob
import re
import cv2
import time
import shutil
import sys


SAVER_NAME = '16x16x16'
RESTORE_VAR = True
SAVE_VAR = False
TRAIN = False
TRAIN_DIV = 1

RUN = True

TEST_ALL = False


MODEL_DIR='inception-2015-12-05'

DATA_FOLDER = 'data'
TRAIN_DIR = 'data_road/training/'
TEST_DIR = 'data_road/testing/image_2' 
Images_DIR = 'Images'
OUT_DIR = './Output'

IMAGE_SIZE = (299, 299)

NUM_CLASS = 2
EPOCHS = 1
LEARN_RATE = 0.0001

TRAIN_DIV = 1

correct_label = tf.placeholder(tf.float32, [IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CLASS])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)


def load_train_data():
    
    images = []
    gt_images = []
#     Load the image
    
    foreground_color = np.array([128, 64, 128])
    
    
    label_paths = {
            re.sub(r'png','jpg',re.sub(r'_train_color', '', os.path.basename(path))): path
            for path in glob(os.path.join('/bdd100k/seg/color_labels/train', '*.png'))}

    files = glob(os.path.join('/bdd100k/seg/images/train', '*.*')) 
    
#    for file_path in files:
    for file_path in files[::TRAIN_DIV]:
        
        gt_image_file = label_paths[os.path.basename(file_path)]

        image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        
        image = cv2.resize(image, IMAGE_SIZE)

#        get gt image
        gt_image_file = label_paths[os.path.basename(file_path)]
        gt_image = cv2.resize(cv2.cvtColor(cv2.imread(gt_image_file), cv2.COLOR_BGR2RGB), IMAGE_SIZE)

        gt_bg = np.all(gt_image != np.array(foreground_color), axis=2)

        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
        
        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
        
        images.append(image)
        gt_images.append(gt_image)  
           
            
    return np.array(images), np.array(gt_images)

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def load_layers():
    
    graph = tf.get_default_graph()
    input_tensor = graph.get_tensor_by_name('DecodeJpeg:0')
    
    mixed2_tensor = graph.get_tensor_by_name('mixed_2/join:0')
    mixed7_tensor = graph.get_tensor_by_name('mixed_7/join:0')
    mixed10_tensor = graph.get_tensor_by_name('mixed_10/join:0')
    pool3_tensor = graph.get_tensor_by_name('pool_3:0')
    
    print(input_tensor)
    print(mixed2_tensor)
    print(mixed7_tensor)     
    print(mixed10_tensor)
    print(pool3_tensor)
    print()
    
    return input_tensor, mixed2_tensor, mixed7_tensor, mixed10_tensor, pool3_tensor

def build_fcn(input_tensor, mixed2_tensor, mixed7_tensor, mixed10_tensor, pool3_tensor):
    
    ks1, ks2, ks3 = SAVER_NAME.split('x')
    
    fcn_conv0 = tf.layers.conv2d(pool3_tensor, filters = NUM_CLASS, kernel_size=1, name="fcn_conv0")
    print(fcn_conv0)
    
    fcn_upsmpl0 = tf.layers.conv2d_transpose(fcn_conv0, filters = mixed7_tensor.get_shape().as_list()[-1],
    kernel_size= int(ks1), strides=(17, 17), padding='same', name="fcn_upsmpl0")
    print(fcn_upsmpl0)
    
    fcn_add0 = tf.add(fcn_upsmpl0, mixed7_tensor, name="fcn_add0")
    print(fcn_add0)
    
    fcn_upsmpl1 = tf.layers.conv2d_transpose(fcn_add0, filters = mixed2_tensor.get_shape().as_list()[-1],
    kernel_size= int(ks2), strides=(2, 2), padding='same', name="fcn_upsmpl1")
    print(fcn_upsmpl1)
    
    fcn_upsmpl1 = tf.pad(fcn_upsmpl1, tf.constant([[0, 0], [1, 0], [0, 1], [0, 0]]), "SYMMETRIC")
    print(fcn_upsmpl1)
    
    fcn_add1 = tf.add(fcn_upsmpl1, mixed2_tensor, name="fcn_add1")
    print(fcn_add1)
    
    fcn_upsmpl2 = tf.layers.conv2d_transpose(fcn_add1, filters = NUM_CLASS,
    kernel_size= int(ks3), strides=(8, 8), padding='same', name="fcn_upsmpl2")
    print(fcn_upsmpl2)
    
    fcn_upsmpl2 = tf.pad(fcn_upsmpl2, tf.constant([[0, 0], [19, 0], [10, 9], [0, 0]]), "SYMMETRIC")
    print(fcn_upsmpl2)
    
    print()
    return fcn_upsmpl2

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
  
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
    
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

    # Calculate distance from actual labels using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
    # Take mean for total loss
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

    # The model implements this operation to find the weights/parameters that would yield correct pixel labels
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

    return logits, train_op, loss_op

    
def train(sess, train_op, cross_entropy_loss, input_tensor, correct_label, keep_prob, learning_rate):

    keep_prob_value = 0.5
    
    train_images, train_gt_images = load_train_data()
    print("========= Training data loaded. =========\n")
#    print("Training images count = ", train_images.shape[0])
    
    for epoch in range(EPOCHS):
        
        total_loss = 0
        img_count = train_images.shape[0]
        for i in range(img_count):

            sys.stdout.write('\r>> Training Image: %d/%d' %(int(i+1),int(img_count)))
                
            loss, _ = sess.run([cross_entropy_loss, train_op],
            feed_dict={input_tensor: train_images[i], correct_label: train_gt_images[i],
            keep_prob: keep_prob_value, learning_rate:LEARN_RATE})
        
            total_loss += loss;
        
        print()
        print("EPOCH {} ...".format(epoch + 1))
        print("Loss = {:.5f}".format(total_loss/img_count))
        print()
        
def run(sess, keep_prob, logits, input_tensor, test_dir):
    
    
    files = glob(test_dir)
    print("Testing image count = ",len(files))
    
    output_dir = os.path.join(OUT_DIR, SAVER_NAME + '.'+ str(time.time())[:10])
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    for file in files:
        
        origin_image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        origin_size = origin_image.shape
                    
        image = cv2.resize(origin_image, IMAGE_SIZE)
                
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, input_tensor: image})
        
        image_out = np.array(im_softmax)
        image_out = image_out[0][0]
        
        image_out = image_out[ :, :, 1].reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])
        segmentation = (image_out > 0.5).reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        mask = scipy.misc.imresize(mask, origin_size)
        
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(origin_image)
        street_im.paste(mask, box=None , mask=mask)
        
        scipy.misc.imsave(os.path.join(output_dir, os.path.basename(file)), street_im)
        
    print('Training Finished. Saving test images to: {}'.format(output_dir))
        
    
def main():
    
    print("\n========= Start! =========\n")
     
    create_graph()
    print("========= Model graph created. =========\n")
    
    input_tensor, mixed2_tensor, mixed7_tensor, mixed10_tensor, pool3_tensor = load_layers()
    print("========= Tensor loaded. =========\n")
    
    fcn_out = build_fcn(input_tensor, mixed2_tensor, mixed7_tensor, mixed10_tensor, pool3_tensor)
    print("========= FCN built. =========\n")
    
    logits, train_op, cross_entropy_loss = optimize(fcn_out, correct_label, learning_rate, NUM_CLASS)
    print("========= Optimized. =========\n")
    
    saver = tf.train.Saver(save_relative_paths=True)
    
    with tf.Session() as sess:
        
#                 Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print("========= Network initialized. =========\n")
        
        if RESTORE_VAR is True:
            saver.restore(sess, os.path.join('save'+SAVER_NAME,'model.ckpt'))
            
        weight = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("Number of weights = ", weight)
        print()
        if TRAIN is True:
            train(sess, train_op, cross_entropy_loss, input_tensor, correct_label, keep_prob, learning_rate)                 
        print("========= Training finished. =========\n")
        
        if SAVE_VAR is True:
            save_path = saver.save(sess, os.path.join('save'+SAVER_NAME,'model.ckpt'))
            print("========= Model saved in path: %s =========\n" % save_path)
  
        print("========= Start testing. =========\n")
        
        if RUN is True:
            if TEST_ALL is True:
                test_dir = os.path.join( DATA_FOLDER, TEST_DIR, '*.*')
            else:
                test_dir = os.path.join( DATA_FOLDER, Images_DIR, '*.*')
            run(sess, keep_prob, fcn_out, input_tensor, test_dir)
        print("\n========= All done. =========\n")
        
#    writer = tf.summary.FileWriter('.')
#    writer.add_graph(tf.get_default_graph())
    


#--------------------------
# MAIN
#--------------------------
if __name__ == '__main__':
    main()
    
    

"""

tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name,'\n')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
"""
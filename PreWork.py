'''
PreWork.py
功能：实现对指定大小的生成图片进行sample与label分类制作
获得神经网络输入的get_files文件，同时为了方便网络的训练，输入数据进行batch处理。
2018/7/19完成
-------copyright@GCN-------
'''
 
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import *
from torchvision import transforms
char0 = []
label_char0 = []
char1 = []
label_char1 = []
char2 = []
label_char2 = []
char3 = []
label_char3 = []
char4 = []
label_char4 = []
char5 = []
label_char5 = []
char6 = []
label_char6 = []
char7 = []
label_char7 = []
char8 = []
label_char8 = []
char9 = []
label_char9 = []
char10 = []
label_char10 = []
char11 = []
label_char11 = []
char12 = []
label_char12 = []
char13 = []
label_char13 = []
char14 = []
label_char14 = []
char15 = []
label_char15 = []
char16 = []
label_char16 = []

def get_file(file_dir):
    # step1：获取路径下所有的图片路径名，存放到
    # 对应的列表中，同时贴上标签，存放到label列表中。
    for file in os.listdir(file_dir + '/0'):
        char0.append(file_dir + '/0' + '/' + file)
        label_char0.append(0)
    for file in os.listdir(file_dir + '/1'):
        char1.append(file_dir + '/1' + '/' + file)
        label_char1.append(1)
    for file in os.listdir(file_dir + '/2'):
        char2.append(file_dir + '/2' + '/' + file)
        label_char2.append(2)
    for file in os.listdir(file_dir + '/3'):
        char3.append(file_dir + '/3' + '/' + file)
        label_char3.append(3)
    for file in os.listdir(file_dir + '/4'):
        char4.append(file_dir + '/4' + '/' + file)
        label_char4.append(4)
    for file in os.listdir(file_dir + '/5'):
        char5.append(file_dir + '/5' + '/' + file)
        label_char5.append(5)
    for file in os.listdir(file_dir + '/6'):
        char6.append(file_dir + '/6' + '/' + file)
        label_char6.append(6)
    for file in os.listdir(file_dir + '/7'):
        char7.append(file_dir + '/7' + '/' + file)
        label_char7.append(7)
    for file in os.listdir(file_dir + '/8'):
        char8.append(file_dir + '/8' + '/' + file)
        label_char8.append(8)
    for file in os.listdir(file_dir + '/9'):
        char9.append(file_dir + '/9' + '/' + file)
        label_char9.append(9)
    for file in os.listdir(file_dir + '/10'):
        char10.append(file_dir + '/10' + '/' + file)
        label_char10.append(10)
    for file in os.listdir(file_dir + '/11'):
        char11.append(file_dir + '/11' + '/' + file)
        label_char11.append(11)
    for file in os.listdir(file_dir + '/12'):
        char12.append(file_dir + '/12' + '/' + file)
        label_char12.append(12)
    for file in os.listdir(file_dir + '/13'):
        char13.append(file_dir + '/13' + '/' + file)
        label_char13.append(13)
    for file in os.listdir(file_dir + '/14'):
        char14.append(file_dir + '/14' + '/' + file)
        label_char14.append(14)
    for file in os.listdir(file_dir + '/15'):
        char15.append(file_dir + '/15' + '/' + file)
        label_char15.append(15)
    for file in os.listdir(file_dir + '/16'):
        char16.append(file_dir + '/16' + '/' + file)
        label_char16.append(16)

    image_list = np.hstack((char0, char1, char2, char3, char4, char5, char6, char7, char8, char9, char10, char11, char12, char13, char14, char15, char16))
    label_list = np.hstack((label_char0, label_char1, label_char2, label_char3, label_char4, label_char5, label_char6, label_char7, label_char8, label_char9, label_char10, label_char11, label_char12, label_char13, label_char14, label_char15, label_char16))
    # 利用shuffle，转置、随机打乱
    temp = np.array([image_list, label_list])  # 转换成2维矩阵
    temp = temp.transpose()    # 转置
    np.random.shuffle(temp)     # 按行随机打乱顺序函数

    all_image_list = list(temp[:, 0])    # 取出第0列数据，即图片路径
    all_label_list = list(temp[:, 1])   # 取出第1列数据，即图片标签
    label_list = [int(i) for i in all_label_list]   # 转换成int数据类型

    return all_image_list, label_list
 
# 将image和label转为list格式数据，因为后边用到的的一些tensorflow函数接收的是list格式数据
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue
    # tf.cast()用来做类型转换
    image = tf.cast(image, tf.string)   # 可变长度的字节数组.每一个张量元素都是一个字节数组
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])   # tf.read_file()从队列中读取图像
 
    # step2：将图像解码，使用相同类型的图像
    image = tf.image.decode_jpeg(image_contents, channels=1)
    # jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    image = tf.image.per_image_standardization(image)
 
    # step4：生成batch

    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=16, capacity=capacity)
 
    # 重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)    # 显示灰度图
    return image_batch, label_batch
    # 获取两个batch，两个batch即为传入神经网络的数据

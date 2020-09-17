'''
Test.py
2019/4/22完成最终网络的测试
-------copyright@GCN-------
'''
# 导入必要的包
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from CNNModel import deep_CNN
 
N_CLASSES = 17
 
img_dir = './DataSet/6/'
log_dir = './final'
lists = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '(', ')', '=']

def get_one_image(img_dir):
    imgs = os.listdir(img_dir)
    img_num = len(imgs)
    # print(imgs, img_num)
    idn = np.random.randint(0, img_num)
    image = imgs[idn]
    image_dir = img_dir + image
    print(image_dir)
    image = Image.open(image_dir)
    #image = tf.image.decode_jpeg(image_contents, channels=3)
    plt.imshow(image)
    plt.show()
    image = image.resize([28, 28])
    image_arr = np.array(image)/256
    return image_arr
    #return image

def test(image_arr):
    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        #image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 28, 28, 1])
        # print(image.shape)
        p = deep_CNN(image, 1, N_CLASSES)
        logits = tf.nn.softmax(p)
        x = tf.placeholder(tf.float32, shape=[28, 28])#, 1])
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            # 调用saver.restore()函数，加载训练好的网络模型
            print('Loading success')
        prediction = sess.run(logits, feed_dict={x: image_arr})
        max_index = np.argmax(prediction)
        print('预测的标签为：', max_index, lists[max_index])
        print('预测的结果为：', prediction)

if __name__ == '__main__':
    img = get_one_image(img_dir)
    test(img)
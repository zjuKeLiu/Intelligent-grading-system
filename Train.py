# 导入文件
import os
import numpy as np
import tensorflow as tf
from PreWork import get_file, get_batch
from CNNModel import deep_CNN, losses, trainning, evaluation
 
# 变量声明
N_CLASSES = 17
IMG_W = 28  # resize图像，太大的话训练时间久
IMG_H = 28
BATCH_SIZE = 40    # 每个batch要放多少张图片
CAPACITY = 20      # 一个队列最大多少
MAX_STEP = 25000
learning_rate = 0.00001  # 一般小于0.0001

# 获取批次batch
train_dir = './DataSet'  # 训练样本的读入路径
logs_train_dir = './final'  #logs存储路径
train, train_label = get_file(train_dir)

train_batch, train_label_batch = get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

train_logits = deep_CNN(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = losses(train_logits, train_label_batch)
train_op = trainning(train_loss, learning_rate)
train_acc = evaluation(train_logits, train_label_batch)

summary_op = tf.summary.merge_all()
sess = tf.Session()
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator() # 设置多线程协调器
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
 
# 进行batch的训练
try:
    # 执行MAX_STEP步的训练，一步一个batch
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

        if step % 100 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
 
        # 保存最后一次网络参数
        checkpoint_path = os.path.join(logs_train_dir, 'thing.ckpt')
        saver.save(sess, checkpoint_path)

        
 
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
 
finally:
    coord.request_stop()
coord.join(threads)
sess.close()
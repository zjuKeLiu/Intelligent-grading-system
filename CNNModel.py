import tensorflow as tf
 
 
# 函数申明
def weight_variable(shape, n):
    initial = tf.truncated_normal(shape, stddev=n, dtype=tf.float32)
    return initial
 
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return initial
 
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
 
def max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
 

def deep_CNN(images, batch_size, n_classes):
    # 第一层卷积
    with tf.variable_scope('conv1') as scope:
        w_conv1 = tf.Variable(weight_variable([3, 3, 1, 64], 1.0), name='weights', dtype=tf.float32)
        b_conv1 = tf.Variable(bias_variable([64]), name='biases', dtype=tf.float32)   # 64个偏置值
        h_conv1 = tf.nn.relu(conv2d(images, w_conv1)+b_conv1, name='conv1')  # 得到28*28*64(假设原始图像是28*28)
    # 第一层池化
    # 3x3最大池化，步长strides为2，池化后执行lrn()操作，

    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = max_pool_2x2(h_conv1, 'pooling1')   # 得到14*14*64
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
 
    # 第二层卷积
    # 32个3x3的卷积核（16通道）
    with tf.variable_scope('conv2') as scope:
        w_conv2 = tf.Variable(weight_variable([3, 3, 64, 32], 0.1), name='weights', dtype=tf.float32)
        b_conv2 = tf.Variable(bias_variable([32]), name='biases', dtype=tf.float32)   # 32个偏置值
        h_conv2 = tf.nn.relu(conv2d(norm1, w_conv2)+b_conv2, name='conv2')  # 得到14*14*32
 
    # 第二层池化
    # 3x3最大池化，步长strides为2
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = max_pool_2x2(h_conv2, 'pooling2')  # 得到7*7*32
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
 
    # 第三层卷积
    # 16个3x3的卷积核（16通道）
    with tf.variable_scope('conv3') as scope:
        w_conv3 = tf.Variable(weight_variable([3, 3, 32, 16], 0.1), name='weights', dtype=tf.float32)
        b_conv3 = tf.Variable(bias_variable([16]), name='biases', dtype=tf.float32)   # 16个偏置值
        h_conv3 = tf.nn.relu(conv2d(norm2, w_conv3)+b_conv3, name='conv3')  # 得到7*7*16
 
    # 第三层池化
    # 3x3最大池化，步长strides为2，池化后执行lrn()操作
    with tf.variable_scope('pooling3_lrn') as scope:
        pool3 = max_pool_2x2(h_conv3, 'pooling3')  # 得到5*5*16
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
 
    # 第四层全连接层
    # 128个神经元，将之前pool层的输出reshape成一行，激活函数relu()
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(norm3, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        w_fc1 = tf.Variable(weight_variable([dim, 128], 0.005),  name='weights', dtype=tf.float32)
        b_fc1 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_fc1 = tf.nn.relu(tf.matmul(reshape, w_fc1) + b_fc1, name=scope.name)
 
    # 第五层全连接层
    # 128个神经元，激活函数relu()
    with tf.variable_scope('local4') as scope:
        w_fc2 = tf.Variable(weight_variable([128 ,128], 0.005),name='weights', dtype=tf.float32)
        b_fc2 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc1, name=scope.name)
 
 
    # 对卷积结果执行dropout操作
    # keep_prob = tf.placeholder(tf.float32)
    h_fc2_dropout = tf.nn.dropout(h_fc2, 0.5)
 
    # Softmax回归层
    # 将前面的FC层输出，做一个线性回归，计算出每一类的得分，在这里是17类，所以这个层输出的是17个得分。
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(weight_variable([128, n_classes], 0.005), name='softmax_linear', dtype=tf.float32)
        biases = tf.Variable(bias_variable([n_classes]), name='biases', dtype=tf.float32)
        softmax_linear = tf.add(tf.matmul(h_fc2_dropout, weights), biases, name='softmax_linear')
        # softmax_linear = tf.nn.softmax(tf.add(tf.matmul(h_fc2_dropout, weights), biases, name='softmax_linear'))
    return softmax_linear
    # 最后返回softmax层的输出

# loss计算
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss
 
# loss损失值优化
# 输入参数：loss。learning_rate，学习速率。
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
 
# 评价/准确率计算
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float16))
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
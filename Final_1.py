'''
导入包
'''
#%%
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import cv2
from math import ceil
import math
from CNNModel import deep_CNN
import tensorflow as tf
'''
必要的函数
'''
# 反相灰度图，将黑白阈值颠倒
def accessPiexl(img):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
       for j in range(width):
           #img[i][j] = 255 - img[i][j]
           img[i][j] = 255 - img[i][j]
    return img

# 反相二值化图像
def accessBinary(img, threshold=128):
    img = accessPiexl(img)
    # 边缘膨胀，不加也可以
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)
    return img

# 根据长向量找出顶点
def extractPeek(array_vals, min_vals=10, min_rect=20):
    extrackPoints = []
    startPoint = None
    endPoint = None
    for i, point in enumerate(array_vals):
        if point > min_vals and startPoint == None:
            startPoint = i
        elif point < min_vals and startPoint != None:
            endPoint = i

        if startPoint != None and endPoint != None:
            extrackPoints.append((startPoint, endPoint))
            startPoint = None
            endPoint = None
    # 剔除一些噪点
    #for point in extrackPoints:
    #    if point[1] - point[0] < min_rect:
    #        extrackPoints.remove(point)
    return extrackPoints

# 寻找边缘，返回边框的左上角和右下角（利用直方图寻找边缘算法（需行对齐））
def findBorderHistogram(path):
    borders = []
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    # 行扫描
    hori_vals = np.sum(img, axis=1)
    hori_points = extractPeek(hori_vals)
    # 根据每一行来扫描列
    for hori_point in hori_points:
        extractImg = img[hori_point[0]:hori_point[1], :]
        vec_vals = np.sum(extractImg, axis=0)
        vec_points = extractPeek(vec_vals, min_rect=0)
        for vect_point in vec_points:
            border = [(vect_point[0], hori_point[0]), (vect_point[1], hori_point[1])]
            borders.append(border)
    return borders

#改变图像为28*28像素点，便于后续处理。
def process_image(img):
    img = np.array(img)
    size = img.shape
    h, w = size[0], size[1]
    min_side = 28
    #长边缩放为min_side 
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    
    # 填充至min_side * min_side
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = int((min_side-new_h)/2), int((min_side-new_h)/2), int((min_side-new_w)/2 + 1), int((min_side-new_w)/2)
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = int((min_side-new_h)/2 + 1), int((min_side-new_h)/2), int((min_side-new_w)/2), int((min_side-new_w)/2)
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = int((min_side-new_h)/2), int((min_side-new_h)/2), int((min_side-new_w)/2), int((min_side-new_w)/2)
    else:
        top, bottom, left, right = int((min_side-new_h)/2 + 1), int((min_side-new_h)/2), int((min_side-new_w)/2 + 1), int((min_side-new_w)/2)
    pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0]) #从图像边界向上,下,左,右扩的像素数目

    pad_img = Image.fromarray(pad_img.astype("uint8"))
    return pad_img

def get_act(x):
    act_vec = []
    for i in x:
        act_vec.append(1/(1+math.exp(-i)))  #激活函数 sigmoid函数
    act_vec = np.array(act_vec)
    return act_vec

Code = '0123456789+-*/()='

#CNN测试
def test(image_arr):
    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        #image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 28, 28, 1])
        # print(image.shape)
        p = deep_CNN(image, 1, 17)
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
        # print('预测的标签为：', max_index, lists[max_index])
        # print('预测的结果为：', prediction)
        return max_index
#%%
def showResults(path, borders, results=None):
    img = cv2.imread(path, 0)
    plt.imshow(img)
    # 绘制
    print(img.shape)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
        if results:
            cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    Imgg = Image.fromarray(img.astype("uint8"))
    Imgg.save('./result.jpg')
    #cv2.imshow('test', img)
    #cv2.waitKey(0)
'''''''''''''''''
主函数开始
'''''''''''''''''
'''废弃ANN模型
#%%
#导入模型
w1 = np.load('w1.npy')
w2 = np.load('w2.npy')
hid_offset = np.load('hid_offset.npy')
out_offset = np.load('out_offset.npy')
'''
img_dir = './'
log_dir = './final'
#%%
#分割图像
path1 = 'data.jpg'
img = cv2.imread(path1, 0)
img = accessBinary(img)
borders = findBorderHistogram(path1)
save_path = './data/'
Img = Image.fromarray(img.astype("float32"))

#%%
equation = []
num = 0
mat_labels = []
mat = []
for x in range(len(borders)):
    ax = (borders[x][0][0],borders[x][0][1],borders[x][1][0],borders[x][1][1])
    image_uint = Img.crop(ax)
    image_uint = process_image(image_uint)
    #image_uint = image_uint.resize((28,28))
    if x>0:
        interval_between = borders[x][0][0] - borders[x-1][1][0]
        interval_all = borders[x][1][0] - borders[x-1][0][0]
        if (borders[x][0][1]!=borders[x-1][0][1])or(interval_between>2*(interval_all - interval_between)):
            num+=1
    image_uint.save('{}/{}.jpg'.format(save_path,x))
    mat_labels.append(num)
    image_test = np.array(image_uint)/256
    index = test(image_test)
    char= Code[index]
    equation.append(char)

#依次识别并保存
#%%
'''
mat = np.array(mat)
mat = mat /256
a = mat.reshape(len(mat),784)
for count in range(len(mat)):
    hid_value = np.dot(a[count], w1) + hid_offset       # 隐层值
    hid_act = get_act(hid_value)                # 隐层激活值
    out_value = np.dot(hid_act, w2) + out_offset             # 输出层值
    out_act = get_act(out_value)                # 输出层激活值
    
    RecognizeResult = np.argmax(out_act)
    char = Code[RecognizeResult]
    equation.append(char)
    print(char)
'''
#%%
ExpectedResult = []
ActualResult = []
i = 0
while i < len(mat_labels):
    CalEqu = []
    Equ = []
    while  i < len(mat_labels)  and equation[i] != '=':
        CalEqu.append(equation[i])
        i += 1
    ThisLabel = mat_labels[i]
    string = "".join(CalEqu)
    number = eval(string)
    ExpectedResult.append(number)
    i += 1
    while i < len(mat_labels) and (mat_labels[i] == ThisLabel):
        Equ.append(equation[i])
        i += 1
    Equ = "".join(Equ)
    ActualResult.append(int(Equ))

#%%
Judge = []
for i in range(len(ActualResult)):
    print(ActualResult[i])
    print(ExpectedResult[i])
    if ActualResult[i] != ExpectedResult[i]:
        Judge.append(int(i))
#%%
if len(Judge) > 0:
    count = 0
    NewBorders = []
    for j in Judge:
        while count < len(mat_labels) and j != mat_labels[count] :
            count += 1
        count1 = count
        while count < len(mat_labels) and j == mat_labels[count] :
            count += 1
        count2 = count
        border_temp = [(borders[count1][0][0],borders[count1][0][1] ),(borders[count2-1][1][0],borders[count2-1][1][1])]
        NewBorders.append(border_temp)
    showResults(path1, NewBorders)

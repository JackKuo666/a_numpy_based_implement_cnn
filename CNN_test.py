# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 16:43:26 2018
# 环境：python 3.6；numpy;pillow
@author: Jack
"""
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import pickle



#---------------------------------show pic--------------------------------------------

def show_pic(pic_data,pic_label,str_data):
    print (str_data+":")
    print (pic_data.shape)
    print (pic_label)
    print (pic_data[:,0:3,0:3][0])
    img0 = Image.fromarray(pic_data[0])
    img1 = Image.fromarray(pic_data[1])
    img2 = Image.fromarray(pic_data[2])
    imgg = Image.merge("RGB",(img0.convert('L'),img1.convert('L'),img2.convert('L')))
    plt.figure()
    plt.imshow(imgg)
    imgg.save("./param/img_"+str_data+".png")
    print ("save pic to: "+"./param/img_"+str_data+".png"+"\ndone!")

#--------------------------------加载处理数据-------------------------------------------
from data_utils import load_CIFAR10
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. 
    """
    # Load the raw CIFAR-10 data
    
    cifar10_dir = './cifar-10-batches-py'

    data_train_all, label_train_all, data_test, label_test = load_CIFAR10(cifar10_dir)
    
    
# --------------------show——image------------------------------------------------------    
    show_pic_data = data_train_all[0]
    show_pic_label = label_train_all[0]
    show_pic(show_pic_data,show_pic_label,"去均值之前的图像")
#---------------------------------------------------------------------------------------    
   
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    data_val = data_train_all[mask]
    label_val = label_train_all[mask]
    mask = range(num_training)
    data_train = data_train_all[mask]
    label_train = label_train_all[mask]
    mask = range(num_test)
    data_test = data_test[mask]
    label_test = label_test[mask]  

    # Normalize the data: subtract the mean image
    mean_image = np.mean(data_train_all, axis=0)
    mean_test_image = np.mean(data_test,axis = 0)
    data_train_m = data_train - mean_image
    data_val_m = data_val - mean_image
    data_test_m = data_test - mean_test_image

    
    return data_train_m, label_train, data_val_m, label_val, data_test_m, label_test

data_train, label_train, data_val, label_val, data_test, label_test = get_CIFAR10_data()


    
test_pic_data = data_train[0]
test_pic_label = label_test[0]
show_pic(test_pic_data,test_pic_label,"去均值之后的图像")



# --------------------------------------set net ------------------------------------------------------
class ConvLayer(object):
    def __init__(self, in_channel, out_channel, kernel_size, lr=0.01, stride = 1, pad = 1, momentum=0.9, reg = 0.75, name='Conv'):
        self.w = np.random.randn(out_channel,in_channel, kernel_size, kernel_size)    #w初值设置，随机产生标准正太分布范围内的数
        self.b = np.random.randn(out_channel)
        #w_shape = (out_channel,in_channel, kernel_size, kernel_size)
        self.layer_name = name
        self.lr = lr
        self.momentum = momentum
        self.stride = stride
        self.pad = pad
        self.reg = reg

        self.prev_gradient_w = np.zeros_like(self.w)
        self.prev_gradient_b = np.zeros_like(self.b)

    def test_forward(self, in_data,param_data):
        self.out = None
        N, C, H, W = in_data.shape
        F, _, HH, WW = self.w.shape                                            #F是out_channel,HH是kernel_size,ww是kernel_size;
        stride, pad = self.stride, self.pad
        H_out = int(1 + (H + 2 * pad - HH) / stride)                           #计算纵向卷积需要滑几步
        W_out = int(1 + (W + 2 * pad - WW) / stride)                           #计算横向卷积需要滑几步
        self.out = np.zeros((N , F , H_out, W_out))

# ---------------test pic add ----------------------
        self.w = param_data[0]
        self.b = param_data[1]
#---------------------------------------------------
        
        in_data_pad = np.pad(in_data, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0) #边缘填充，这里上下左右各填一个像素
        for i in range(H_out):
            for j in range(W_out):
                in_data_pad_masked = in_data_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
                for k in range(F):
                    self.out[:, k , i, j] = np.sum(in_data_pad_masked * self.w[k, :, :, :], axis=(1,2,3))+ self.b[k] #注意：这里比原版加了一个+ self.b[k]

        self.bottom_val = in_data
        return self.out



class FCLayer:
    def __init__(self, in_num, out_num, lr=0.01, momentum=0.9, std = 1e-4, reg = 0.75):
        self._in_num = in_num
        self._out_num = out_num
        self.w =std * np.random.randn(in_num, out_num)
        self.b =std * np.zeros(out_num)
        self.lr = lr
        self.momentum = momentum
        self.prev_grad_w = np.zeros_like(self.w)
        self.prev_grad_b = np.zeros_like(self.b)
        self.reg = reg

    def test_forward(self, in_data,param_data):
        
# ---------------test pic add ----------------------
        self.w = param_data[0]
        self.b = param_data[1]
#---------------------------------------------------
        
        self.bottom_val = in_data
        self.top_val = in_data.dot(self.w) + self.b                               #这里虽然b.shape是1x10，但是在相加的时候是：矩阵A的每一行都加上向量b
        return self.top_val





class ReLULayer:
    def __init__(self, name='ReLU'):
        pass

    def test_forward(self, in_data,param_data):
        #不需要param_data  
        in_data[in_data<0]
        self.top_val = in_data
        return in_data




class MaxPoolingLayer:
    def __init__(self, kernel_size, stride = 1, name='MaxPool'):
        self.kernel_size = kernel_size
        self.stride = stride

    def test_forward(self, in_data,param_data):
        self.bottom_val = in_data
        #不需要param_data
        N, C, H, W = in_data.shape
        HH, WW, stride = self.kernel_size, self.kernel_size, self.stride       #HH和WW均为kernel_size
        H_out = int((H - HH) / stride + 1)                                     #计算纵向需要滑几步
        W_out = int((W - WW) / stride + 1)                                     #计算横向需要滑几步
        out = np.zeros((N, C, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                x_masked = in_data[:, :, i * stride: i * stride + HH, j * stride: j * stride + WW]
                out[:, :, i, j] = np.max(x_masked, axis=(2, 3))
        return out





class FlattenLayer:
    def __init__(self, name='Flatten'):
        pass

    def test_forward(self, in_data,param_data):
        #不需要param_data
        self.in_batch, self.in_channel, self.r, self.c = in_data.shape
        return in_data.reshape(self.in_batch, self.in_channel * self.r * self.c)


class SoftmaxLayer:
    def __init__(self, name='Softmax'):
        pass

    def test_forward(self, in_data,param_data):
        #不需要param_data
        shift_scores = in_data - np.max(in_data, axis=1).reshape(-1, 1)                    #在每行中10个数都减去该行中最大的数字
        self.top_val = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)
        return self.top_val

    
    
# -------------------------------------定义Net----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Net:
    def __init__(self):                              #定义layers
        self.layers = []

    def addLayer(self, layer):                       
        self.layers.append(layer)


    
    def test_pic(self, data, label,data_all_param_pkl):
        lay_num = len(self.layers)
        in_data = data
        for i in range(lay_num):
            out_data = self.layers[i].test_forward(in_data,data_all_param_pkl[i])
            in_data = out_data
        out_idx = np.argmax(in_data, axis=1)   #找出softmax层输出[1000,10]每行概率最大的位置
        
        label_idx = label                      #label存储的是[1000,]每个图片对应的标签在one-hot中位置
        print ("该图正确的标签是："+str(label_idx))
        print ("预测到该图的标签是："+str(out_idx))
        print ("正确率为："+str(np.sum(out_idx == label_idx) / float(out_idx.shape[0])))
        

#----------------------------------定义网络--------------------------------------------

rate = 1e-5
net = Net()
net.addLayer(ConvLayer(3, 32, 5, rate))
net.addLayer(MaxPoolingLayer(3,2))
net.addLayer(ReLULayer())

net.addLayer(ConvLayer(32, 16, 3, rate))
net.addLayer(MaxPoolingLayer(3,2))
net.addLayer(ReLULayer())

net.addLayer(FlattenLayer())
net.addLayer(FCLayer(6 * 6 * 16, 100, rate))
net.addLayer(ReLULayer())
net.addLayer(FCLayer(100, 10, rate))
net.addLayer(SoftmaxLayer())
print('net build ok')
#--------------------------------------------------------------------------------------



# ----------------------------load param data ----------------------------------------

all_param_pkl_file = open('./param/all_param.pkl', 'rb')

data_all_param_pkl = pickle.load(all_param_pkl_file)

all_param_pkl_file.close()

#-------------------------------------------------------------------------------------



test_pic_data_reshape = test_pic_data.reshape(1,3,32,32)
net.test_pic(test_pic_data_reshape,test_pic_label,data_all_param_pkl)






















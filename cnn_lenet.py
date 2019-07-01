# -*- coding: utf-8 -*-
# 环境：python 3.6；numpy;pillow
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import pickle


def conv2(X, k):
    # as a demo code, here we ignore the shape check
    x_row, x_col = X.shape
    k_row, k_col = k.shape
    ret_row, ret_col = x_row - k_row + 1, x_col - k_col + 1
    ret = np.empty((ret_row, ret_col))
    for y in range(ret_row):
        for x in range(ret_col):
            sub = X[y: y + k_row, x: x + k_col]
            ret[y, x] = np.sum(sub * k)
    return ret



def padding(in_data, size):
    cur_r, cur_w = in_data.shape[0], in_data.shape[1]
    new_r = cur_r + size * 2
    new_w = cur_w + size * 2
    ret = np.zeros((new_r, new_w))
    ret[size:cur_r + size, size:cur_w + size] = in_data
    return ret


def discreterize(in_data, size):
    num = in_data.shape[0]
    ret = np.zeros((num, size))
    for i, idx in enumerate(in_data):
        ret[i, idx] = 1
    return ret


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

    def forward(self, in_data):
        self.out = None
        N, C, H, W = in_data.shape
        F, _, HH, WW = self.w.shape                                            #F是out_channel,HH是kernel_size,ww是kernel_size;
        stride, pad = self.stride, self.pad
        H_out = int(1 + (H + 2 * pad - HH) / stride)                           #计算纵向卷积需要滑几步
        W_out = int(1 + (W + 2 * pad - WW) / stride)                           #计算横向卷积需要滑几步
        self.out = np.zeros((N , F , H_out, W_out))

        in_data_pad = np.pad(in_data, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0) #边缘填充，这里上下左右各填一个像素
        for i in range(H_out):
            for j in range(W_out):
                in_data_pad_masked = in_data_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
                for k in range(F):
                    self.out[:, k , i, j] = np.sum(in_data_pad_masked * self.w[k, :, :, :], axis=(1,2,3))+ self.b[k] #注意：这里比原版加了一个+ self.b[k]

        self.bottom_val = in_data
        return self.out

    def backward(self, residual):
        N, C, H, W = self.bottom_val.shape
        F, _, HH, WW = self.w.shape
        stride, pad = self.stride, self.pad
        H_out = int(1 + (H + 2 * pad - HH) / stride)
        W_out = int(1 + (W + 2 * pad - WW) / stride)

        x_pad = np.pad(self.bottom_val, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
        dx = np.zeros_like(self.bottom_val)
        
        dw = np.zeros_like(self.w)
        
        db = np.sum(residual, axis=(0, 2, 3))                #============# 这个db是将top_diff的维度[10 16 14 14]中0,2,3维相加得到

    # 这个是原作，对dw,dx计算用的点乘法============================================================================

        dx_pad = np.zeros_like(x_pad)
        for i in range(H_out):
            for j in range(W_out):
                x_pad_masked = x_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
                for k in range(F):  # compute dw
                    dw[k, :, :, :] += np.sum(x_pad_masked * (residual[:, k, i, j])[:, None, None, None], axis=0)  
                    # dw=pad（bottom_data）* top_diff_ij
                for n in range(N):  # compute dx_pad
                    dx_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += np.sum((self.w[:, :, :, :] * (residual[n, :, i,j])[:, None, None, None]), axis=0)
                    # 这个dx = (w)* （top_diff_ij）
                    
        if pad == 0:
            dx[:,:,:,:] = dx_pad[:, :, :, :]
        else :
            dx[:,:,:,:] = dx_pad[:, :, pad:-pad, pad:-pad]
            

     # =======================================================================================================
     

     # 这个是我的方法分别对dx,dw计算，对dw计算用的卷积法==============================================================

#        rot_w = np.rot90(self.w,2,(2,3))    #第一个2是逆时针旋转90度2次，也就是180度。（2,3）表示旋转con_w的第（2,3）维。 
#        pad_diff_H = HH - (1 + pad)
#        pad_diff_W = WW - (1 + pad)
#        residual_pad = np.pad(residual, ((0,), (0,), (pad_diff_H,), (pad_diff_W,)), mode='constant', constant_values=0)  
#        #注意：这里由于反向传播计算出来的卷积核大小是5的时候反向传播pad需要设置成3，有规律摸索，构成公式
#       
#        for i in range(H_out):      #========# 计算dx
#            for j in range(W_out):
#                residual_pad_masked = residual_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
#                for h in range(C):
#                    dx[:, h , i, j] = np.sum(residual_pad_masked[:,:,:,:] * rot_w[:, h, :, :], axis=(1,2,3))    
#        
#        for m in range(HH):         #========# 这个是dw=pad（bottom_data）卷积 top_diff
#            for n in range(WW):
#                x_pad_masked_d = x_pad[:, :, m * stride:m * stride + H_out, n * stride:n * stride + W_out]
#                for k in range(F):
#                    for p in range(C):
#                        dw[k, p, m, n] = np.sum(x_pad_masked_d[:,p,:,:] * residual[:, k, :, :], axis=(0,1,2)) 

     # ====================================================================================                        

        self.w -= self.lr * (dw + self.prev_gradient_w * self.reg)
        self.b -= self.lr * db
        self.prev_gradient_w = self.w
        return dx



    def save_param(self,):
        return self.w, self.b





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

    def forward(self, in_data):
        self.bottom_val = in_data
        self.top_val = in_data.dot(self.w) + self.b                               #这里虽然b.shape是1x10，但是在相加的时候是：矩阵A的每一行都加上向量b
        return self.top_val

    def backward(self, loss):
        residual_x = loss.dot(self.w.T)
        self.w -= self.lr * (self.bottom_val.T.dot(loss) + self.prev_grad_w * self.reg)
        self.b -= self.lr * (np.sum(loss, axis=0))                                          #axis = 0表示列相加
        self.prev_grad_w = self.w
        self.prev_grad_b = self.b
        return residual_x


    def save_param(self,):
        return self.w, self.b





class ReLULayer:
    def __init__(self, name='ReLU'):
        pass

    def forward(self, in_data):

        in_data[in_data<0] = 0
        self.top_val = in_data
        return in_data

    def backward(self, residual):
        return (self.top_val > 0) * residual                                    # (self.top_val > 0)表示大于0的为1，不大于0的为0；为relu对输入导数

    def save_param(self,):
        return np.arange(9).reshape(3,3), np.arange(9).reshape(3,3)       #自己瞎写的，目的是随便存俩数组，占位用



class MaxPoolingLayer:
    def __init__(self, kernel_size, stride = 1, name='MaxPool'):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, in_data):
        self.bottom_val = in_data

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

    def backward(self, residual):
        N, C, H, W = self.bottom_val.shape
        HH, WW, stride = self.kernel_size, self.kernel_size, self.stride
        H_out = int((H - HH) / stride + 1)
        W_out = int((W - WW) / stride + 1)
        dx = np.zeros_like(self.bottom_val)

        for i in range(H_out):
            for j in range(W_out):
                x_masked = self.bottom_val[:, :, i * stride: i * stride + HH, j * stride: j * stride + WW]
                max_x_masked = np.max(x_masked, axis=(2, 3))
                temp_binary_mask = (x_masked == (max_x_masked)[:, :, None, None])
                dx[:, :, i * stride: i * stride + HH, j * stride: j * stride + WW] += temp_binary_mask * (residual[:, :, i,j])[:, :,None, None]
        return dx


    def save_param(self,):
        return np.arange(9).reshape(3,3), np.arange(9).reshape(3,3)       #自己瞎写的，目的是随便存俩数组，占位用




class FlattenLayer:
    def __init__(self, name='Flatten'):
        pass

    def forward(self, in_data):
        self.in_batch, self.in_channel, self.r, self.c = in_data.shape
        return in_data.reshape(self.in_batch, self.in_channel * self.r * self.c)

    def backward(self, residual):
        return residual.reshape(self.in_batch, self.in_channel, self.r, self.c)
    
    def save_param(self,):
        return np.arange(9).reshape(3,3), np.arange(9).reshape(3,3)  #自己瞎写的，目的是随便存俩数组，占位用


class SoftmaxLayer:
    def __init__(self, name='Softmax'):
        pass

    def forward(self, in_data):
        shift_scores = in_data - np.max(in_data, axis=1).reshape(-1, 1)                    #在每行中10个数都减去该行中最大的数字
        self.top_val = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)
        return self.top_val

    def backward(self, residual):
        N = residual.shape[0]
        dscores = self.top_val.copy()
        dscores[range(N), list(residual)] -= 1                                           #loss对softmax层的求导
        dscores /= N
        return dscores
    
    def save_param(self,):
        return np.arange(9).reshape(3,3), np.arange(9).reshape(3,3)  #自己瞎写的，目的是随便存俩数组，占位用

    
    
# -------------------------------------定义Net----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Net:
    def __init__(self):                              #定义layers
        self.layers = []

    def addLayer(self, layer):                       
        self.layers.append(layer)

    def train(self, trainData, trainLabel, validData, validLabel, batch_size, iteration):
        train_num = trainData.shape[0]
        strainData = trainData
        strainLabel = trainLabel
#--------------------- eval可视化(1/2) ----------------------------------------------------------------  
        eval_list = np.zeros(iteration)
        
#-------------------------------------------------------------------------------------------------     
        for iter in range(iteration):
            index = np.random.choice([ i for i in range(train_num)], train_num)   #将traindata的1000个图像顺序打乱
            # index = [i for i in range(train_num)]
            trainData = strainData[index]
            trainLabel = strainLabel[index]                                       #将traindata按乱序重新排列


            if iter > 100:
                lay_num = len(self.layers)
                for i in range(lay_num):
                   self.layers[i].lr *= (0.001 ** ( iter - 100 ) / 100)

            print(str(time.clock()) + '  iter=' + str(iter))                     #打印出这个iter开始时间以及iter次数

#--------------------Loss可视化(1/2)-------------------------------------------------------------------
#            loss_list = np.zeros(1000)
#            ii = 0
#--------------------------------------------------------------------------------------------------

            for batch_iter in range(0, train_num, batch_size):
                if batch_iter + batch_size < train_num:
                    loss = self.train_inner(trainData[batch_iter: batch_iter + batch_size],
                                     trainLabel[batch_iter: batch_iter + batch_size])
                else:
                    loss = self.train_inner(trainData[batch_iter: train_num],
                                     trainLabel[batch_iter: train_num])
                print(str(batch_iter) + '/' + str(train_num) + '   loss : ' + str(loss))
                
# ----------------------loss可视化(1/2)-----------------------------------------------------------------                
#                loss_list[ii] = loss
#                ii =ii+1
#                plt.figure()
#                plt.plot(loss_list, "r-")
#                plt.xlabel("Batches")
#                plt.ylabel("Loss")
#                plt.grid()
#                plt.show()
#--------------------------------------------------------------------------------------------------                
                
                
            print(str(time.clock()) + "  eval=" + str(self.eval(trainData, trainLabel)))
            print(str(time.clock()) + "  eval=" + str(self.eval(validData, validLabel)))
#--------------------- eval可视化(1/2) ----------------------------------------------------------------  
            eval_list[iter] = self.eval(validData, validLabel)
            plt.figure()
            plt.plot(eval_list, "r-")
            plt.xlabel("iterations")
            plt.ylabel("Correct_rate")
            plt.grid()
            plt.show()

#-----------------------------当训练达到设置param的时候，保存参数，结束训练------------------------------

            if iter == 10:
                self.save_all_param()
                sys.exit(0)
                
    def save_all_param(self,):
        all_param = []
        lay_num = len(self.layers)
        file_pkl = open("./data_pic/all_param.pkl",'wb') 
        for i in range(lay_num):
            out_data = self.layers[i].save_param()
            all_param.append(out_data)          
        pickle.dump(all_param, file_pkl)
        file_pkl.close()
        print ("paramter is save in:"+"./data_pic/all_param.pkl"+"done!!!"+"\n paramter size is : "+str(len(all_param)))

#----------------------------------------------------------------------------------------------------        


    def train_inner(self, data, label):
        lay_num = len(self.layers)                                       #返回层的个数
        in_data = data
      #  print ("网络层数以及每层的网络shape:")
        for i in range(lay_num):
            out_data = self.layers[i].forward(in_data)
            in_data = out_data                                           #训练结果
      #      print (out_data.shape)
     #   print ("每层参数的shape:")
        N = out_data.shape[0]
        loss = -np.sum(np.log(out_data[range(N), list(label)]))          #训练结果与标签的loss函数
        loss /= N
        residual_in = label
        for i in range(0, lay_num):
            residual_out = self.layers[lay_num - i - 1].backward(residual_in)
            residual_in = residual_out
       #     print(residual_out.shape)
        

        
        return loss

    def eval(self, data, label):
        lay_num = len(self.layers)
        in_data = data
        for i in range(lay_num):
            out_data = self.layers[i].forward(in_data)
            in_data = out_data
        out_idx = np.argmax(in_data, axis=1)   #找出softmax层输出[1000,10]每行概率最大的位置
        # label_idx = np.argmax(label, axis=1)
        label_idx = label                      #label存储的是[1000,]每个图片对应的标签在one-hot中位置
        return np.sum(out_idx == label_idx) / float(out_idx.shape[0])    #正确率=预测正确的图片数量/总数量


#----------------------------------定义网络--------------------------------------------

rate = 1e-5
net = Net()
net.addLayer(ConvLayer(3, 6, 5, rate,1,0))  #(in_channel, out_channel, kernel_size, lr=0.01, stride = 1, pad = 1,)
net.addLayer(ReLULayer())
net.addLayer(MaxPoolingLayer(2,2))          #(self, kernel_size, stride = 1, name='MaxPool')


net.addLayer(ConvLayer(6, 16, 5, rate,1,0))
net.addLayer(ReLULayer())
net.addLayer(MaxPoolingLayer(2,2))

net.addLayer(FlattenLayer())
net.addLayer(FCLayer(5 * 5 * 16, 120, rate))  #(self, in_num, out_num, lr=0.01, momentum=0.9, std = 1e-4, reg = 0.75)
net.addLayer(ReLULayer())
net.addLayer(FCLayer(120, 84, rate))
net.addLayer(ReLULayer())
net.addLayer(FCLayer(84, 10, rate))
net.addLayer(SoftmaxLayer())
# print('net build ok')

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

#-----------------------训练---------------------------------------------------
data_train, label_train, data_val, label_val, data_test, label_test = get_CIFAR10_data()
N = 1000
M = 100
net.train(data_train[0:N], label_train[0:N], data_val[0:M], label_val[0:M],10,100)

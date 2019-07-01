# -*- coding: utf-8 -*-
#依赖：python3.x ；numpy；Pillow
import pickle
import numpy as np



def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  print ("load data: "+filename)
  with open(filename, 'rb') as f:
    datadict = pickle.load(f,encoding='iso-8859-1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):                       #读取data_batch_1到data_batch_5的数据
    f = 'data_batch_%d' % (b)
    X, Y = load_CIFAR_batch(ROOT+"/"+f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(ROOT+"/"+'test_batch')  #读取test_batch的数据
  return Xtr, Ytr, Xte, Yte


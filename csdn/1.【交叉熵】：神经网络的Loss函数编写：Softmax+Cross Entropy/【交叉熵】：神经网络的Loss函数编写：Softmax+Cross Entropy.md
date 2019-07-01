# 1.交叉熵的作用
通过神经网络解决多分类问题时，最常用的一种方式就是在最后一层设置n个输出节点，无论在浅层神经网络还是在CNN中都是如此，比如，在AlexNet中最后的输出层有1000个节点： 
![这里写图片描述](https://img-blog.csdn.net/20180724103325706?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/0)

一般情况下，最后一个输出层的节点个数与分类任务的目标数相等。假设最后的节点数为N，那么对于每一个样例，神经网络可以得到一个N维的数组作为输出结果，数组中每一个维度会对应一个类别。在最理想的情况下，如果一个样本属于k，那么这个类别所对应的的输出节点的输出值应该为1，而其他节点的输出都为0，即[0,0,1,0,….0,0]，这个数组也就是样本的Label，是神经网络最期望的输出结果，**交叉熵就是用来判定实际的输出与期望的输出的接近程度！**

# 2. 为什么使用交叉熵？
[一文搞懂交叉熵在机器学习中的使用，透彻理解交叉熵背后的直觉](https://blog.csdn.net/tsyccnh/article/details/79163834)这篇文字章介绍了什么是交叉熵？交叉熵在机器学习中的应用的例子。

看完交叉熵的概念之后，相信你对交叉熵有一个大致的了解。至于为什么使用**交叉熵代价函数**，而不使用**方差代价函数**作为深度学习反向传播的代价函数？下面首先简单介绍一下[**深度学习反向传播算法。**](https://blog.csdn.net/u014313009/article/details/51039334)

>**反向传播算法（Back propagation）**是目前用来训练人工神经网络（Artificial Neural Network，ANN）的最常用且最有效的算法。其主要思想是：
（1）将训练集数据输入到ANN的输入层，经过隐藏层，最后达到输出层并输出结果，这是ANN的前向传播过程；
（2）由于ANN的输出结果与实际结果有误差，则计算估计值与实际值之间的误差，并将该误差从输出层向隐藏层反向传播，直至传播到输入层；
（3）在反向传播的过程中，根据误差调整各种参数的值；不断迭代上述过程，直至收敛。

更多反向传播算法介绍可以访问：[反向传播算法（过程及公式推导）](https://blog.csdn.net/u014313009/article/details/51039334)。看完反向传播算法之后，你大概可以知道代价函数的用处了。那么代价函数在这里有两种计算方式：

1.**方差代价函数**


<div align=center>
![这里写图片描述](https://img-blog.csdn.net/201807291731055?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

2.**交叉熵代价函数**

<div align=center>
![这里写图片描述](https://img-blog.csdn.net/20180729173229534?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

那么为什么大家都使用交叉熵代价函数，而不使用方差代价函数呢？

答：**它可以克服方差代价函数更新权重过慢的问题**

　　我们可以从下图看到方差代价函数的梯度含有sigmoid函数的导数。
<div align=center>　
![这里写图片描述](https://img-blog.csdn.net/20180729174202947?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/0)

　　而从下图是交叉熵代价函数的梯度，可以看到没有导数项，所以在计算的时候速度很快。
<div align=center>　
![这里写图片描述](https://img-blog.csdn.net/20180729174949656?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/0)

具体的原因，可以看：
[交叉熵代价函数（cross-entropy cost function）](https://blog.csdn.net/wtq1993/article/details/51741471)
[交叉熵代价函数（作用及公式推导）](https://blog.csdn.net/u014313009/article/details/51043064)
这两篇文章都是介绍二次代价函数的不足，以及为什么使用交叉熵代价函数。


# 3.Softmax回归处理

神经网络的原始输出不是一个概率值，实质上只是输入的数值做了复杂的加权和与非线性处理之后的一个值而已，那么如何将这个输出变为概率分布？

这就是Softmax层的作用，假设神经网络的原始输出为y1,y2,….,yn，那么经过Softmax回归处理之后的输出为： 


<div align=center>
![这里写图片描述](https://img-blog.csdn.net/20180724103520756?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/0)

 很显然的是： 
 
<div align=center>
 ![这里写图片描述](https://img-blog.csdn.net/20180724103543182?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/0)

  **Softmax回归处理就是：使单个节点的输出变成的一个概率值，经过Softmax处理后结果作为神经网络最后的输出。**


# 4.交叉熵的原理

交叉熵刻画的是实际输出（概率）与期望输出（概率）的距离，也就是交叉熵的值越小，两个概率分布就越接近。假设概率分布p为期望输出，概率分布q为实际输出，H(p,q)为交叉熵，则：


<div align=center>
  ![这里写图片描述](https://img-blog.csdn.net/20180724103753252?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/0)
  
 这个公式如何表征距离呢，举个例子：
假设N=3，期望输出为p=(1,0,0)，实际输出q1=(0.5,0.2,0.3)，q2=(0.8,0.1,0.1)，那么：

<div align=center>
![这里写图片描述](https://img-blog.csdn.net/20180724103820815?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/0)

很显然，q2与p更为接近，它的交叉熵也更小。

除此之外，交叉熵还有另一种表达形式，还是使用上面的假设条件：

<div align=center>
![这里写图片描述](https://img-blog.csdn.net/20180724103907353?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/0)

其结果为：

<div align=center>
![这里写图片描述](https://img-blog.csdn.net/20180724103944759?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/0)


以上的所有说明针对的都是单个样例的情况，而在实际的使用训练过程中，数据往往是组合成为一个batch来使用，所以对用的神经网络的输出应该是一个m*n的二维矩阵，其中m为batch的个数，n为分类数目，而对应的Label也是一个二维矩阵，还是拿上面的数据，组合成一个batch=2的矩阵：

<div align=center>
![这里写图片描述](https://img-blog.csdn.net/20180724104119158?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/0)


 所以交叉熵的结果应该是一个列向量（根据第一种方法）：
  
<div align=center>
 ![这里写图片描述](https://img-blog.csdn.net/20180724104145220?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/0)
 
  而对于一个batch，最后取平均为0.2。[^1]

# 5.代码
代码见我的github仓库：[csdn/ Softmax-Cross Entropy](https://github.com/JackKuo666/csdn/tree/master/Softmax-Cross%20Entropy)

## 5.1.softmax
```
# import numpy as np
a = np.array([[-0.69314718 ,-1.60943791, -1.2039728],[-0.22314355, -2.30258509, -2.30258509]])
print ("数组a的值为：\n" + str(a))
print ("找出a中每行最大值：")
print (np.max(a, axis=1).reshape(-1,1))
b = a - np.max(a, axis=1).reshape(-1, 1)
print ("a中每行均减去本行最大值后的数组b：")
print (b)
a_softmax = np.exp(a) / np.sum(np.exp(a), axis=1).reshape(-1, 1)
print ("对数组a进行softmax：")
print (a_softmax)
b_softmax = np.exp(b) / np.sum(np.exp(b), axis=1).reshape(-1, 1)
print ("对去掉最大值的进行softmax：")
print (b_softmax)
```
## 5.2.cross entropy
```
print ("我们上次得到的softmax为：")
print (b_softmax)
d = np.log10(b_softmax)                         # log下什么都不写默认是自然对数e为底 ，np.log10()是以10为底
print ("对softmax取ln：")
print (d)
print ("找出softmax中每行我们标签概率最大的两个数，也就是第一行的第0个，第二行的第0个：")
print (b_softmax[range(2), list([0,0])])
c = np.log10(b_softmax[range(2), list([0,0])])
print ("分别对这两个数进行ln：")
print (c)
print ("最后，因为这两行是一个batch的两个，所以，加和去平均，得到的就是Loss：")
print (-np.sum(np.log10(b_softmax[range(2), list([0,0])]))*(1/2))
```
[^1]:[理解交叉熵作为损失函数在神经网络中的作用](https://blog.csdn.net/chaipp0607/article/details/73392175)
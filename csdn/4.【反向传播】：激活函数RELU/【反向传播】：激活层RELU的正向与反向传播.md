# 1.神经网络激活函数介绍
## 1.1 为什么使用激活函数
首先摆结论，因为线性模型的表达能力不够，**引入激活函数是为了添加非线性因素**。（知乎回答[^1]解释了这句话。）


**激活函数通常有如下一些性质：** [^2]

 - **非线性：**当激活函数是线性的时候，一个两层的神经网络就可以逼近基本上所有的函数了。但是，如果激活函数是恒等激活函数的时候（即f(x)=x），就不满足这个性质了，而且如果MLP使用的是恒等激活函数，那么其实整个网络跟单层神经网络是等价的。
 
 - **可微性：**当优化方法是基于梯度的时候，这个性质是必须的。
 
 - **单调性：**当激活函数是单调的时候，单层网络能够保证是凸函数。
 
 - **激活函数输出约等于输入【f(x)≈x】：**当激活函数满足这个性质的时候，如果参数的初始化是random的很小的值，那么神经网络的训练将会很高效；如果不满足这个性质，那么就需要很用心的去设置初始值。
 
 **- 输出值的范围：**当激活函数输出值是**有限**的时候，基于梯度的优化方法会更加 稳定，因为特征的表示受有限权值的影响更显著；当激活函数的输出是**无限**的时候，模型的训练会更加高效，不过在这种情况小，一般需要更小的learning rate。

## 1.2 常用的激活函数

该部分需要一篇专门的文章[【卷积神经网络专题】：6.激活函数](https://blog.csdn.net/weixin_37251044/article/details/81334562)来介绍，这里先引用其他博主的介绍：

各个激活函数的用的地方：https://blog.csdn.net/leo_xu06/article/details/53708647
各个激活函数的优缺点：https://blog.csdn.net/Jaster_wisdom/article/details/78380839

# 2.Relu函数的正向传播

## 2.1 RELU 的图像如下：


<center>![这里写图片描述](https://img-blog.csdn.net/20180731221057515?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/0)</center>

## 2.2 RELU 的公式如下：

$$Relu(x) = 
\left\{  
             \begin{array}{lr}  
             x, &  x>0\\  
             0, & x\leq0 
             \end{array}  
\right.$$

## 2.3 RELU正向传播的python-numpy代码如下
```
in_data[in_data<0] = 0
```
# 3.Relu函数的反向传播
## 3.2 反向传播公式：
　　其在x=0处是不可微的，但是在深度学习框架的代码中为了解决这个直接将其在x=0处的导数置为1，所以它的导数也就变为了 ：
　　$$\delta_{Relu(x)} = 
\left\{  
             \begin{array}{lr}  
             1, &  x>0\\  
             0, & x\leq0 
             \end{array}  
\right.$$

## 3.3反向传播python-numpy代码：
```
return (self.top_val > 0) * residual                                    
# (self.top_val > 0)表示大于0的为1，不大于0的为0；为relu对输入导数

```


# 4.Relu的优点

 >1. relu的稀疏性（激活函数的作用）；
 >2. 还是防止梯度弥散（也叫作梯度消失，是针对sigmoid函数这一缺点的）；[^3]
 >3. 加快计算（正反向传播代码好实现）
 
 下面展开叙述：[^4]
 
 首先我们看下sigmoid和relu的曲线

![这里写图片描述](https://img-blog.csdn.net/20180731222756200?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/0)

然后可以得到sigmoid的导数

![这里写图片描述](https://img-blog.csdn.net/20180731222858991?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/0)

以及relu的导数

![这里写图片描述](https://img-blog.csdn.net/20180731222911348?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/0)

结论:

　　1.sigmoid的导数只有在0附近的时候有比较好的激活性,在正负饱和区的梯度都接近于0,所以这会造成梯度弥散,而relu函数在大于0的部分梯度为常数,所以不会产生梯度弥散现象。
　　
　　2.relu函数在负半区的导数为0 ,所以一旦神经元激活值进入负半区,那么梯度就会为0,也就是说这个神经元不会经历训练,即所谓的稀疏性。 
　　
　　3.relu函数的导数计算更快,程序实现就是一个if-else语句,而sigmoid函数要进行浮点四则运算。










[^1]:[神经网络激励函数的作用是什么？有没有形象的解释？](https://www.zhihu.com/question/22334626)

[^2]:[【机器学习】神经网络-激活函数-面面观(Activation Function)](https://blog.csdn.net/cyh_24/article/details/50593400)

[^3]:[神经网络为什么要使用激活函数，为什么relu要比sigmoid要好](https://blog.csdn.net/piaodexin/article/details/77163297)

[^4]:[深度学习中，为何Relu激活函数相比sigmoid具备稀疏激活性，这个结论是怎样得到的呢？](https://www.zhihu.com/question/52020211/answer/152378276)

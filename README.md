# a_numpy_based_implement_cnn

这是我的博客[《不用框架，使用Python搭建基于numpy的卷积神经网络来进行cifar-10分类的深度学习系统》](https://blog.csdn.net/weixin_37251044/article/details/81290728)的代码实现。
该代码有两个主要部分和其他组成：
>训练

>测试

>其他：卷积层可视化


依赖：

>Python3.6
>numpy;pillow;scipy;matplotlib

IED:
>spyder;jupyter notebook;

其中训练部分由两个Python文件和一个文件夹组成：
```
data_utils.py
cnn.py
./cifar-10-batches-py$ ls
batches.meta  data_batch_2  data_batch_4  test_batch
data_batch_1  data_batch_3  data_batch_5
```
[注：cifar-10-batches-py文件夹存放的是cifar—10数据集，相应的`data_utils.py`Python文件是提取数据集的代码。]


测试部分由两个Python文件和一个文件夹组成：
```
data_utils.py
CNN_test.py 
./param$ ls
all_param.pkl  diff_9.npy                             param_10.npy
diff_0.npy     img_0.png                              param_1.npy
diff_10.npy    img_data_1_con1[10-32-30-30].png       param_2.npy
diff_1.npy     img_data_2_maxpool1[10-32-14-14].png   param_3.npy
diff_2.npy     img_data_3_relu1.png                   param_4.npy
diff_3.npy     img_data_4_con2[10-16-14-14].npy.png   param_5.npy
diff_4.npy     img_data_5_maxpooling2[10-16-6-6].png  param_6.npy
diff_5.npy     img_data_6_relu2[10-16-6-6].png        param_7.npy
diff_6.npy     img_去均值之后的图像.png               param_8.npy
diff_7.npy     img_去均值之前的图像.png               param_9.npy
diff_8.npy     param_0.npy
```
[注：param文件夹中存放的是训练好的wangluo参数]

运行代码：
1.训练：
```
python cnn.py 
```
输出：

2.测试：
```
python CNN_test.py
```
输出：
```
load data: ./cifar-10-batches-py/data_batch_1
load data: ./cifar-10-batches-py/data_batch_2
load data: ./cifar-10-batches-py/data_batch_3
load data: ./cifar-10-batches-py/data_batch_4
load data: ./cifar-10-batches-py/data_batch_5
load data: ./cifar-10-batches-py/test_batch
去均值之前的图像:
(3, 32, 32)
6
[[ 59.  43.  50.]
 [ 16.   0.  18.]
 [ 25.  16.  49.]]
save pic to: ./param/img_去均值之前的图像.png
done!
去均值之后的图像:
(3, 32, 32)
3
[[ -71.71074  -87.14036  -81.05044]
 [-114.0993  -129.3446  -112.2169 ]
 [-104.72472 -112.71662  -80.47348]]
save pic to: ./param/img_去均值之后的图像.png
done!
net build ok
该图正确的标签是：3
预测到该图的标签是：[6]
正确率为：0.0
```









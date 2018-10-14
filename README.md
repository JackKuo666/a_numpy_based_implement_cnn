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














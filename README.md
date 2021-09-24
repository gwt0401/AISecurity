本项目主要由训练 LENET 识别 CIFAR-10 数据集，以及使用多项式回归拟合某河堤水位数据

acc.txt:里面记录了训练LENET时，14个EPOCH的准确度

CIFAR-10_Classify.py是训练LENET的代码

ex5data1.mat是某河堤水位数据

HyperparameterTuning.py是多项式回归拟合某河堤水位数据的代码

------------------------------------------------------------------------------------------

CIFAR-10_Classify.py中使用Pytorch构建网络模型

HyperparameterTuning.py中使用多项式回归拟合数据，调整正则项参数lamda，以在训练完后的模型上测试测试集的损失最小



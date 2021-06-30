#Sigmoid曲线:
import matplotlib.pyplot as plt
import numpy as np

def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

x= np.arange(-10, 10, 0.1)
h = Sigmoid(x)            #Sigmoid函数
plt.plot(x, h)
plt.axvline(0.0, color='k')   #坐标轴上加一条竖直的线（0位置）
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
plt.axhline(y=0.5, ls='dotted', color='k')  #在y=0.5的地方加上黑色虚线
plt.yticks([0.0,  0.5, 1.0])  #y轴标度
plt.ylim(-0.1, 1.1)       #y轴范围
plt.show()


#读入数据文件
attributes=['SepalLength','SepalWidth','PetalLength','PetalWidth'] #鸢尾花的四个属性名

datas=[]
labels=[]

# with open('IRIS_dataset.txt','r') as f:
#     for line in f:
#         linedata=line.split(',')
#         datas.append(linedata[:-1]) #前4列是4个属性的值
#         labels.append(linedata[-1].replace('\n','')) #最后一列是类别

#读入数据集的数据：
data_file=open('IRIS_dataset.txt','r')
for line in data_file.readlines():
    # print(line)
    linedata = line.split(',')
    # datas.append(linedata[:-1])  # 前4列是4个属性的值(误判的样本的个数为：7
    datas.append(linedata[:-3])  # 前2列是2个属性的值(误判的样本的个数为：30
    labels.append(linedata[-1].replace('\n', ''))  # 最后一列是类别

datas=np.array(datas)
datas=datas.astype(float) #将二维的字符串数组转换成浮点数数组
labels=np.array(labels)
kinds=list(set(labels)) #3个类别的名字列表


# LogisticRegression算法，训练数据，传入参数为数据集（包括特征数据及标签数据），结果返回训练得到的参数 W
def LogRegressionAlgorithm(datas,labels):
    kinds = list(set(labels))  # 3个类别的名字列表
    means=datas.mean(axis=0) #各个属性的均值
    stds=datas.std(axis=0) #各个属性的标准差
    N,M= datas.shape[0],datas.shape[1]+1  #N是样本数，M是参数向量的维
    K=3 #k=3是类别数

    data=np.ones((N,M))
    data[:,1:]=(datas-means)/stds #对原始数据进行标准差归一化

    W=np.zeros((K-1,M))  #存储参数矩阵
    priorEs=np.array([1.0/N*np.sum(data[labels==kinds[i]],axis=0) for i in range(K-1)]) #各个属性的先验期望值

    liklist=[]
    for it in range(1000):
        lik=0 #当前的对数似然函数值
        for k in range(K-1): #似然函数值的第一部分
            lik -= np.sum(np.dot(W[k],data[labels==kinds[k]].transpose()))
        lik +=1.0/N *np.sum(np.log(np.sum(np.exp(np.dot(W,data.transpose())),axis=0)+1)) #似然函数的第二部分
        liklist.append(lik)

        wx=np.exp(np.dot(W,data.transpose()))
        probs=np.divide(wx,1+np.sum(wx,axis=0).transpose()) # K-1 *N的矩阵
        posteriorEs=1.0/N*np.dot(probs,data) #各个属性的后验期望值
        gradients=posteriorEs - priorEs +1.0/100 *W #梯度，最后一项是高斯项，防止过拟合
        W -= gradients #对参数进行修正
    print("输出W为：",W)
    return W


#根据训练得到的参数W和数据集，进行预测。输入参数为数据集和由LogisticRegression算法得到的参数W，返回值为预测的值
def predict_fun(datas,W):
    N, M = datas.shape[0], datas.shape[1] + 1  # N是样本数，M是参数向量的维
    K = 3  # k=3是类别数
    data = np.ones((N, M))
    means = datas.mean(axis=0)  # 各个属性的均值
    stds = datas.std(axis=0)  # 各个属性的标准差
    data[:, 1:] = (datas - means) / stds  # 对原始数据进行标准差归一化

    # probM每行三个元素，分别表示data中对应样本被判给三个类别的概率
    probM = np.ones((N, K))
    print("data.shape:", data.shape)
    print("datas.shape:", datas.shape)
    print("W.shape:", W.shape)
    print("probM.shape:", probM.shape)
    probM[:, :-1] = np.exp(np.dot(data, W.transpose()))
    probM /= np.array([np.sum(probM, axis=1)]).transpose()  # 得到概率

    predict = np.argmax(probM, axis=1).astype(int)  # 取最大概率对应的类别
    print("输出predict为：", predict)
    return predict


# 1.确定坐标轴范围，x，y轴分别表示两个特征
x1_min, x1_max = datas[:, 0].min(), datas[:, 0].max()  # 第0列的范围
x2_min, x2_max = datas[:, 1].min(), datas[:, 1].max()  # 第1列的范围
x1, x2 = np.mgrid[x1_min:x1_max:150j, x2_min:x2_max:150j]  # 生成网格采样点，横轴为属性x1，纵轴为属性x2
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
#.flat 函数将两个矩阵都变成两个一维数组，调用stack函数组合成一个二维数组
print("grid_test = \n", grid_test)

grid_hat = predict_fun(grid_test,W)  # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
#grid_hat本来是一唯的，调用reshape()函数修改形状，将其grid_hat转换为两个特征（长度和宽度）
print("grid_hat = \n", grid_hat)
print("grid_hat.shape: = \n", grid_hat.shape) # (150, 150)


# 3.绘制图像
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

alpha = 0.5

plt.pcolormesh(x1, x2, grid_hat, cmap=plt.cm.Paired)  # 预测值的显示
# 调用pcolormesh()函数将x1、x2两个网格矩阵和对应的预测结果grid_hat绘制在图片上
# 可以发现输出为三个颜色区块，分布表示分类的三类区域。cmap=plt.cm.Paired/cmap=cm_light表示绘图样式选择Paired主题
# plt.scatter(datas[:, 0], datas[:, 1], c=labels, edgecolors='k', s=50, cmap=cm_dark)  # 样本
plt.plot(datas[:, 0], datas[:, 1], 'o', alpha=alpha, color='blue', markeredgecolor='k')
##绘制散点图
plt.scatter(datas[:, 0], datas[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
plt.xlabel(u'花萼长度', fontsize=13)  #X轴标签
plt.ylabel(u'花萼宽度', fontsize=13)  #Y轴标签
plt.xlim(x1_min, x1_max) # x 轴范围
plt.ylim(x2_min, x2_max) # y 轴范围
plt.title(u'鸢尾花LogisticRegression二特征分类', fontsize=15)
# plt.legend(loc=2)  # 左上角绘制图标
# plt.grid()
plt.show()
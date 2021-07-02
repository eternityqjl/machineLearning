import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

path = r'C:\Users\83621\Documents\vscode\machineLearning\endDataClassify\data\A训练集'
 
files = os.listdir(path)

path_A = path + str('\\') + str('0')
path_B = path + str('\\') + str('1')
path_C = path + str('\\') + str('2')
dirA = os.listdir(path_A)
dirB = os.listdir(path_B)
dirC = os.listdir(path_C)

def read_file(path,dirr,label):
    shape = np.array(dirr).shape[0]
    sample = pd.read_csv(path+'\\'+dirr[0],header=None)
    dirlist = np.ones((shape,800 + 1))
    for i in range(shape):
        dirlist[i,0] = label
        data = pd.read_csv(path+'\\'+dirr[i],header=None)
        data = np.array(data).reshape(1,800)
        dirlist[i,1:] = data
    return dirlist
    
    
datasetA = read_file(path_A, dirA,0)
datasetB = read_file(path_B, dirB,1)
datasetC = read_file(path_C, dirC,2)

dataset = np.vstack((datasetC,datasetB))
dataset = np.vstack((dataset,datasetA))
pd.DataFrame(dataset).to_csv(path + 'test_dataset1.csv')
# 给出训练集数据以及对应的类别
#-------------------------------加载csv数据集----------------------------------------------------------------------------
def load_csv_data(filename):
    data_raw = pd.read_csv(filename)
    data_raw1 = np.array(data_raw)

    group = data_raw1[:,2:]     #属性值
    labels = np.array(data_raw1[:,1], dtype = int)    #标签值

    print('该组数据的每个样本共有%d种属性'%(group.shape[1]))
    print('该组样本共有3个标签')
    return group, labels
#-----------------------------------------------------------------------------------------------------------

# 使用KNN算法进行分类
def classify(test,dataSet,label,k):
    dataSize = dataSet.shape[0]
    
    #计算欧氏距离
    diff = np.tile(test,(dataSize,1)) - dataSet
    sqdiff = diff ** 2
    squareDist = sqdiff.sum(axis = 1)   #将行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5
    
    ## 对距离进行排序：argsort()根据元素的值从大到小对元素进行排序，返回下标
    sortedDistIndex = np.argsort(dist)
    
    classCount={}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        
        # K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    # 选取出现的类别次数最多的类别
    maxCount = 0
    for key,value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key
    return classes

if __name__ == "__main__":
    filename = r'C:\Users\83621\Documents\vscode\machineLearning\endDataClassify\data\A训练集test_dataset1.csv'
    x,y = load_csv_data(filename)
    pca = PCA(n_components=50)
    x = pca.fit_transform(x)     #等价于pca.fit(X) pca.transform(X)
    x = pca.inverse_transform(x)  #将降维后的数据转换成原始数据
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 25)
    y_test1 = []
    for i in x_test:
        y_test1.append(classify(i,x_train,y_train,10))
    y_test2 = list(y_test)
    v = list(map(lambda x: x[0]-x[1], zip(y_test1, y_test2)))
    judgelabel = v.count(0)
    length = y_test.shape[0]
    accuracy = judgelabel/length
    print('正确率：',accuracy)

    print(1)
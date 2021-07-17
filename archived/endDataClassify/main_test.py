import os
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets,tree

dataPath_train_0 = r'C:\Users\83621\Documents\vscode\machineLearning\endDataClassify\data\A训练集\0'
dataPath_train_1 = r'C:\Users\83621\Documents\vscode\machineLearning\endDataClassify\data\A训练集\1'
dataPath_train_2 = r'C:\Users\83621\Documents\vscode\machineLearning\endDataClassify\data\A训练集\2'

dataPath_val_0 = r'C:\Users\83621\Documents\vscode\machineLearning\endDataClassify\data\B分类和聚类数据准备-测试学生版本\0'
dataPath_val_1 = r'C:\Users\83621\Documents\vscode\machineLearning\endDataClassify\data\B分类和聚类数据准备-测试学生版本\1'
dataPath_val_2 = r'C:\Users\83621\Documents\vscode\machineLearning\endDataClassify\data\B分类和聚类数据准备-测试学生版本\2'

files = os.listdir(dataPath_train_0)

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


def load_data(dataPath):
    files = os.listdir(dataPath)
    dataSet = list()
    for each_train_csv_0 in files:
        data_raw = pd.read_csv(os.path.join(dataPath, each_train_csv_0))
        data_raw.values.tolist()
        dataSet.append(data_raw)
        dataSet = np.array(dataSet)
    return dataSet

class KNearestNeighbor(object):
   def __init__(self):
       pass

   # 训练函数
   def train(self, X, y):
       self.X_train = X
       self.y_train = y

   # 预测函数
   def predict(self, X, k=1):
       # 计算L2距离
       num_test = X.shape[0]
       num_train = self.X_train.shape[0]
       dists = np.zeros((num_test, num_train))    # 初始化距离函数
       # because(X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train, so
       d1 = -2 * np.dot(X, self.X_train.T)    # shape (num_test, num_train)
       d2 = np.sum(np.square(X), axis=1, keepdims=True)    # shape (num_test, 1)
       d3 = np.sum(np.square(self.X_train), axis=1)    # shape (1, num_train)
       dist = np.sqrt(d1 + d2 + d3)
       # 根据K值，选择最可能属于的类别
       y_pred = np.zeros(num_test)
       for i in range(num_test):
           dist_k_min = np.argsort(dist[i])[:k]    # 最近邻k个实例位置
           y_kclose = self.y_train[dist_k_min]     # 最近邻k个实例对应的标签
           y_pred[i] = np.argmax(np.bincount(y_kclose.tolist()))    # 找出k个标签中从属类别最多的作为预测类别

       return y_pred

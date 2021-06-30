import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from collections import defaultdict
from sklearn.model_selection import train_test_split

def load_data():
    #加载数据, 返回数据和类别值
    data = load_iris()
    return data['data'], data['target']

def prob(element,arr):
        #计算元素在列表中出现的概率
        prob = 0.0
        for a in arr:
            if element == a:
                prob += 1/len(arr)
        if prob == 0.0:
            prob = 0.001
        return prob

def step(arr,n):
        #按一组属性值的最大最小值差将属性分为n阶,修改属性值为对应阶数
        ma = max(arr)
        mi = min(arr)
        for i in range(len(arr)):
            for j in range(n):
                a = mi + (ma-mi)*(j/n)
                b = mi + (ma-mi)*((j+1)/n)
                if arr[i] >= a and arr[i] <= b:
                    arr[i] = j+1
                    break
        return arr

def preprocess(x):
        #因为不同特征的数值集大小相差巨大，造成部分概率矩阵变得稀疏，需要进行数据分割
        for i in range(x.shape[1]):
            x[:,i] = step(x[:,i],5)
        return x

if __name__ == "__main__":
    x,y = load_data()
    #划分数据集和训练集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.5,random_state = 100)
    x_train = preprocess(x_train)

    for yi in y_train:   #遍历每种类型
            for i in range(x.shape[1]): #遍历所有样本
                sample = x_train[y_train==yi, i]    #类别yi下的所有样本
                #计算该列的概率分布
                pxy = [prob(xi,sample) for xi in self.x[i]]
                self.pxy[yi][i] = pxy
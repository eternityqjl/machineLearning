import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from collections import defaultdict
from sklearn.model_selection import train_test_split
import csv
import re
import os
import pandas as pd

def load_csv_data(filename):
    data_raw = pd.read_csv(filename)
    data_raw1 = np.array(data_raw)

    number = len(data_raw1)     #样本个数
    data = data_raw1[1:,1:]     #属性值
    target = np.array(data_raw1[1:,0], dtype = int)    #标签值

    print('该组数据的每个样本共有%d种属性'%(data.shape[1]))
    print('该组样本共有2个标签')
    print('该组数据共有%d个样本'%number)
    return data, target

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

def preprocess(x, n):
    #因为不同特征的数值集大小相差巨大，造成部分概率矩阵变得稀疏，需要进行数据分割
    for i in range(x.shape[1]): #将每个属性的数据根据n进行分割
        x[:,i] = step(x[:,i],n)
    return x

filename = r'C:\Users\83621\Documents\vscode\machineLearning\patientClassify\theta_gamma_power_spectrum.csv'

if __name__ == "__main__":
    n = 4
    x,y = load_csv_data(filename)
    x1 = preprocess(x, n)
    x2 = x1.T

    for xi in x2:
        mask = np.unique(x2)
        tmp = []
        for v in mask:
            tmp.append(np.sum(xi==v))
        ts = np.max(tmp)
        max_v = mask[np.argmax(tmp)]
        print(f'这个值：{max_v}出现的次数最多，为{ts}次')







            


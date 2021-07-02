# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:17:44 2021

@author: Aaron
"""


import os 
import pandas as pd
import numpy as np
import math
import random
from sklearn import preprocessing
path = r'D:\SourceCode\Python Source Code\MachineLearning\Untitled Folder'
from sklearn.model_selection import train_test_split



def readdata(path):
    dataset = pd.read_csv(path+'\\'+'dataset.csv')
    return dataset


def readtest(path):
    dataset = pd.read_csv(path+'\\'+'test_dataset.csv')
    return dataset

def classify(feature,label):
    #分类函数
    #把不同类别的分出来
    Ar = feature[np.where(label == 0)]
    Br = feature[np.where(label == 1)]
    A_mean = datasetMean(Ar)
    B_mean = datasetMean(Br)
    A_std = datasetStd(Ar)
    B_std = datasetStd(Br)
    mathlog = np.reshape(np.array([A_mean,A_std,B_mean,B_std]),(4,32))
    return mathlog,Ar,Br
            
def load_data(dataset):
    train_feature = dataset[:,1:]
    train_label = dataset[:,0]
    return train_feature,train_label

def load_data1(dataset):
    #random.shuffle(dataset)

    train_feature = dataset[:,1:]
    train_label = dataset[:,0]
    return train_feature,train_label

def load_data_rate(dataset,rate):
    row, col = np.array(dataset.shape)
    random.shuffle(dataset)
    train_feature = dataset[0:int(rate * row),1:]
    train_label = dataset[0:int(rate * row),0]
    test_feature = dataset[int(rate * row):,1:]
    test_label = dataset[int(rate * row):,0]
    return train_feature,train_label,test_feature,test_label

def normalize1(dataset):
    row,col = np.array(dataset.shape)
    data = np.zeros(dataset.shape)
    for i in range(row):
        for j in range(col):
            data[i,j] = (dataset[i,j]-min(dataset[:,j]))/(max(dataset[:,j])-min(dataset[:,j]))
    return data

def datasetMean(dataset):
    #求解平均值
    return sum(dataset)/float(len(dataset))
def datasetStd(dataset):
    #求解均方差标注差
    return np.std(dataset,axis=0)

def normalize2(dataset):
    row,col = np.array(dataset.shape)
    mean = datasetMean(dataset)
    std = datasetStd(dataset)
    data = np.zeros(dataset.shape)
    for i in range(row):
        for j in range(col):
            data[i,j] = (dataset[i,j]-mean[j])/(std[j])
    return data

def calcDist(x1, x2):
    '''
    计算两个样本点向量之间的距离
    使用的是欧氏距离，即 样本点每个元素相减的平方  再求和  再开方
    欧式举例公式这里不方便写，可以百度或谷歌欧式距离（也称欧几里得距离）
    :param x1:向量1
    :param x2:向量2
    :return:向量之间的欧式距离
    '''
    return np.sqrt(np.sum(np.square(x1 - x2)))
    #return np.sum(np.square(x1 - x2))
    #马哈顿距离计算公式
    #return np.sum(x1 - x2)

def distance(dataset):
    row,col = np.array(dataset.shape)
    dis = np.zeros((row,row))
    for i in range(row):
        for j in range(row):
            dis[i,j] = calcDist(dataset[i], dataset[j])
    return dis

def getclost(dataset,K):
    topKList = np.argsort(np.array(dataset),axis = 1)[:,:K]        #升序排序
    row,col = np.array(dataset.shape)
    libellist = np.zeros((row,2))
    for i in range(row):
        for index in topKList[i,:]:
            libellist[i,int(train_label[int(index)])] += 1
    return libellist

def getclost1(dataset,K):
    topKList = np.argsort(np.array(dataset))[:,:K]        #升序排序
    row,col = np.array(dataset.shape)
    libellist = np.zeros((row,2))
    for i in range(row):
        for index in topKList[i,:]:
            libellist[i,int(train_label[int(index)])] += 1
    return libellist
    
def getLabel(dataset):
    row,col = np.array(dataset.shape)
    label = np.zeros((row,))
    for i in range(row):
        if dataset[i,0] > dataset[i,1]:
            label[i] = 0
        else:
            label[i] = 1
    return label

def test(test_feature,train_feature):
    row,_ = np.array(test_feature.shape)
    col,_ = np.array(train_feature.shape)
    Distance = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            Distance[i,j] = calcDist(test_feature[i,:], train_feature[j,:])
    return Distance

def getAcc(real,calc):
    row = np.array(real.shape)
    Acc = np.zeros((row[0],))
    Acc = real[:,0] - calc[:,0]
    return Acc


def readdata(path):
    dataset = pd.read_csv(path+'\\'+'dataset.csv')
    return dataset

if __name__ == "__main__":
    print('start read transSet')
    dataset = np.array(readdata(path))[:,1:]
    test_dataset = np.array(readtest(path))[:,1:]
    print('end read transSet')

    labels = dataset[:,0] - 1
    features = dataset[:,1:]
    #train_feature,test_feature, train_label, test_label =train_test_split(features,labels,test_size=0.0, random_state=0)
    #train_feature = preprocessing.scale(train_feature)
    #test_feature = preprocessing.scale(test_feature)
    train_feature = preprocessing.scale(features)
    train_label = labels
    row = np.array(train_label.shape)[0]
    
    test_labels = test_dataset[:,0] - 1
    test_features = test_dataset[:,1:]
    test_feature = preprocessing.scale(test_features)
    test_label = test_labels
    test_row = np.array(test_label.shape)[0]
    

    
    dist = distance(features)
    libellist = getclost(dist, 10)
    getlabel = getLabel(libellist)
    Acc = sum(abs(train_label - getlabel))
    print("准确率：%d%s" % (int((1-Acc/row)*100),"%"))
    print("训练集：")
    for i in range(50):
        libellist = getclost(dist, i)
        getlabel = getLabel(libellist)
        Acc = sum(abs(train_label - getlabel))
        
        print("%d%s准确率：%d%s" % (i,':',int((1-Acc/row)*100),"%"))
        
    Distance = test(test_feature,train_feature)
    print("测试集：")
    for i in range(50):
        libellist = getclost(Distance, i)
        getlabel = getLabel(libellist)
        Acc = sum(abs(test_label - getlabel))
        print("%d%s准确率：%d%s" % (i,':',int((1-Acc/test_row)*100),"%"))
    '''
    for i in range(260):
        if acc[i] > 0:
            acc[i] = 0
        else:
            acc[i] = 1
    for i in range(260,180):
        if acc[i] > 0:
            acc[i] = 1
        else:
            acc[i] = 0
    accuracy = 1 - sum(acc)/180
    print("准确率：%d%s" % (int(accuracy*100),"%"))
    '''
    '''
    data_test = pd.read_excel(current_file_path +'\\'+ 'theta_gamma_power_spectrum_test.xlsx')#,index_col = '序号'
    data_test = np.array(data_test)
    test_feature,test_label = load_data1(data_test)
    train_feature = preprocessing.scale(train_feature)
    test_feature = preprocessing.scale(test_feature)
    Dist = test(test_feature,train_feature)
    Libellist = getclost1(Dist, 2)
    acc = Libellist[:,0] - Libellist[:,1]
    err = 0
    for i in range(np.array(test_feature.shape)[0]):
        if Libellist[i,0] > Libellist[i,1]:
            acc[i] = 0
        elif Libellist[i,0] < Libellist[i,1]:
            acc[i] = 1
    Acc = sum(abs(acc - test_label))
    print("准确率：%d%s" % (int((1-Acc/20)*100),"%"))
    '''

    
            



    

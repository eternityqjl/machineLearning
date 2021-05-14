import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from collections import defaultdict
from sklearn.model_selection import train_test_split
"""
iris datasets:
    共有150个样本
    4种属性：'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
    3个类别：'setosa' 'versicolor' 'virginica'
"""

def load_data():
    #加载数据, 返回数据和类别值
    data = load_iris()
    return data['data'], data['target']

"""
python class的使用:
每个函数的参数都需要包含关键词self,或者换成其他词也可以
"""
class NBClassifier():
    def __init__(self):
        #类的构造函数__init__(), self代表类的实例
        self.y = [] #类别集合
        self.x = [] #每种属性的数值集合
        self.py = defaultdict(float)    #类别的概率分布
        """
        pxy以字典的形式储存了:
        P(a1|y1) P(a2|y2) ... P(am|y1)
        P(a1|y2) P(a2|y2) ... P(am|y2)
        ...
        P(a1|yn) P(a2|yn) ... P(am|yn)
        """
        self.pxy = defaultdict(dict)    #每个类别下每个属性的概率分布
        self.n = 5

    def prob(self,element,arr):
        #计算元素element在列表arr中出现的概率
        prob = 0.0
        for a in arr:
            if element == a:
                prob += 1/len(arr)
        if prob == 0.0:
            prob = 0.001
        return prob

    def get_set(self,x,y):
        self.y = list(set(y))   #创建一个无序不重复元素集, 即获得所有标签类型
        for i in range(x.shape[1]): #shape[1]代表了x的列数
            self.x.append(list(set(x[:,i])))#记录下每一列的数值集,即第i个样本每种属性的所有值
    
    def step(self,arr,n):
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

    def preprocess(self,x):
        #因为不同特征的数值集大小相差巨大，造成部分概率矩阵变得稀疏，需要进行数据分割
        for i in range(x.shape[1]):
            x[:,i] = self.step(x[:,i],self.n)
        return x
    
    def score(self,x,y):
        y_test = self.predict(x)
        score = 0.0
        for i in range(len(y)):
            if y_test[i] == y[i]:
                score += 1/len(y)
        return score

    def predict(self,samples):
        #预测函数
        samples = self.preprocess(samples)  #samples为所有样本
        y_list = []
        for m in range(samples.shape[0]):
            yi = self.predict_one(samples[m,:])
            y_list.append(yi)
        return np.array(y_list)

    def predict_one(self,x):
        #预测单个样本
        max_prob = 0.0
        max_yi = self.y[0]
        for yi in self.y:
            prob_y = self.py[yi]
            for i in range(len(x)): #遍历每个样本的4个属性值
                prob_x_y = self.pxy[yi][i][self.x[i].index(x[i])]#p(xi|y)
                prob_y *= prob_x_y#计算p(x1|y)p(x2|y)...p(xn|y)p(y)
            if prob_y > max_prob:
                max_prob = prob_y
                max_yi = yi
        return max_yi

    def fit(self,x,y):
        #通过训练集x_train和y_train训练模型
        x = self.preprocess(x)
        self.get_set(x, y)
        #计算P(y_i)
        for yi in self.y:   #遍历每类标签yi
            self.py[yi] = self.prob(yi, y)  #yi在y中出现的概率
        #计算P(x|y)
        for yi in self.y:   #遍历每种类型,共3种
            for i in range(x.shape[1]): #遍历每种属性的属性值,共4种
                sample = x[y==yi, i]    #第i列中所有标签为yi的样本的属性值集合
                #计算该列的概率分布
                pxy = [self.prob(xi,sample) for xi in self.x[i]]    #每种属性共有5个属性值(已经归一化)
                self.pxy[yi][i] = pxy   #
        print("train score", self.score(x, y))  #通过训练集预测模型正确的概率




if __name__ == "__main__":
    x,y = load_data()
    #划分数据集和训练集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.5,random_state = 100)
    clf = NBClassifier()
    clf.fit(x_train,y_train)
    score = clf.score(x_test,y_test)    #通过测试机验证模型正确率
    print('test score',score)
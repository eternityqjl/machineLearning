import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#-------------------------------加载csv数据集----------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------------------------------------

class KNearestNeighbor(object):
    def __init__(self):
        pass

    # 训练函数
    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    # 预测函数
    def predict(self, X, k):
        # 计算L2距离
        #测试集X: (37,32), 训练集X_train: (160,32)
        num_test = X.shape[0]    #测试集数量
        num_train = self.X_train.shape[0]    #训练集数量
        dists = np.zeros((num_test, num_train))    # 初始化距离函数
        # because(X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train, so
        d1 = -2 * np.dot(X, self.X_train.T)    # shape (num_test, num_train)
        d2 = np.sum(np.square(X), axis=1, keepdims=True)    # shape (num_test, 1)
        d3 = np.sum(np.square(self.X_train), axis=1)    # shape (1, num_train)
        dist = np.sqrt(d1 + d2 + d3)     #(37,160)
        # 根据K值，选择最可能属于的类别
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            dist_k_min = np.argsort(dist[i])[:k]    # 最近邻k个实例位置
            y_kclose = self.y_train[dist_k_min]     # 最近邻k个实例对应的标签
            y_pred[i] = np.argmax(np.bincount(y_kclose.tolist()))    # 找出k个标签中从属类别最多的作为预测类别

        return y_pred

if __name__ == "__main__":
    train_data = r'C:\Users\83621\Documents\vscode\machineLearning\patientClassify\theta_gamma_power_spectrum_01.csv'
    test_data = r'C:\Users\83621\Documents\vscode\machineLearning\patientClassify\theta_gamma_power_spectrum_test.csv'
    x,y = load_csv_data(train_data)
    X_test0,y_test0 = load_csv_data(test_data)
    
    X_train = np.vstack([x])
    y_train = np.hstack([y])

    X_test = np.vstack([X_test0])
    y_test = np.hstack([y_test0])

    #X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state = 100)

    KNN = KNearestNeighbor()
    KNN.train(X_train, y_train)
    y_pred = KNN.predict(X_test, 12)
    accuracy = np.mean(y_pred == y_test)
    print('测试集预测准确率：%f' % accuracy)
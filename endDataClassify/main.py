import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets,tree

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



if __name__ == "__main__":
    train_data = r'C:\Users\83621\Documents\vscode\machineLearning\patientClassify\theta_gamma_power_spectrum.csv'
    #test_data = r''
    x,y = load_csv_data(train_data)
    #x_test,y_test = load_csv_data(test_data)

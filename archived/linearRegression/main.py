from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

from generateTrainData import geneData

#Cost Function
#y = b + mx
#points is the set of the Train data
#point[0]是x，point[1]是y
#
#代价函数costFunction
def costFunction(b, m, points):
    for point in points:
        ErrorTotal += ((b + m*point[0]) - point[1]) ** 2
    return ErrorTotal / (2 * float(len(points)))

def stepGradient(b_current, m_current, b_gradient, m_gradient, points, learningRate):
    N = float(len(points))
    for point in points:
        x = point[0]
        y = point[1]
        b_gradient += (2/N) * ((b_current + m_current * x) - y)
        m_gradient += (2/N) * x * ((b_current + m_current * x) - y)
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m, b_gradient, m_gradient]

if __name__ == '__main__':
    iterations = 100    #迭代次数
    learningRate = 0.0001   #学习率，决定了下降的步伐大小
    points = geneData() #生成训练集
    b = 0   #线性方程参数b,m的初始值
    m = 0   
    b_gradient = 0  #代价函数梯度下降结果的初始值
    m_gradient = 0
    for i in list(range(1, iterations+1)):  #循环进行梯度下降，求得结果
        b,m,b_gradient,m_gradient = stepGradient(b,m,b_gradient,m_gradient,points,learningRate)

    for point in points:    #画出样本点
        plt.scatter(point[0], point[1])

    #画出得到的直线
    t = np.arange(-3,3,0.01)
    s = b + m * t
    plt.scatter(t,s,linewidths=0.5)
    plt.show()

    #输出结果
    print("b=%f"%b)
    #print(b)
    print("m=%f"%m)
from sklearn.datasets import make_regression

#generate the train data
def geneData():
    points = []
    xSet, ySet = make_regression(n_samples=100, n_features=1, n_targets=1, noise=20)
    for x,y in zip(xSet,ySet):
        x=x[0]
        point = [x,y]
        points.append(point)
    return points

"""
if __name__ == '__main__':
    points = geneData()
    print(points)
"""
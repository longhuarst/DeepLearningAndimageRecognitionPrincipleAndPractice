import numpy as np
import matplotlib.pyplot as plt

##给出训练数据以及对应的类别
def createDataSet():
    group = np.array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5],[1.1,1.0],[0.5,1.5]])
    labels = np.array(['A','A','B','B','A','B'])
    return group,labels
if __name__=='__main__':
    group,labels = createDataSet()
    plt.scatter(group[labels=='A',0],group[labels=='A',1],color = 'r', marker='*')#对于类别为A的数据集我们使用红色六角形表示
    plt.scatter(group[labels=='B',0],group[labels=='B',1],color = 'g', marker='+')#对于类别为B的数据集我们使用绿色十字形表示
    plt.show()

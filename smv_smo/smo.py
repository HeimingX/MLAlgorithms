# -*- coding: utf-8 -*-
'''
Created on 2016年5月1日

@author: heiming
'''
import numpy as np
from scipy import io
import matplotlib.pyplot as plt

class Params:
    def __init__(self):
        ##alpha:lagrange multiplyer
        self.a = []
        ##b bias
        self.b = 0.0
        ## punish parameter
        self.c = 10
        ## decision error
        self.epsilon = 0.001
        ## stop tolerance
        self.stopTol = 0.0001
        ## smo difference of predict accuracy
        self.e = []
        ## maximum iteration
        self.maxIter = 100
        ## kernel type
        self.kernelType = 'gaussian'
        ## gaussian parameter
        self.sigma = 0.5
    
    def setC(self, c):
        self.c = c
    
    def setEpsilon(self, epsilon):
        self.epsilon = epsilon
        
    def setKernel(self, kernelType, *args):
        self.kernelType = kernelType
        if self.kernelType == 'gaussian':
            self.sigma = args[0]
    
    def setMaxIter(self, Maxiter):
        self.maxIter = Maxiter

def loadData(filePath, fileType):
    """
    load样本
    """
    if fileType == 'txt':
        data = np.loadtxt(filePath)
        trainData = data[:, :-1]
        trainLabel = data[:, -1]
        testData = []
        testLabel = []
    if fileType == 'mat':
        data = io.loadmat(filePath)
        trainData = data['xapp']
        trainLabel = np.reshape(data['yapp'], newshape = (1, data['yapp'].size))[0]
        testData = data['xtest']
        testLabel = np.reshape(data['ytest'], newshape = (1, data['ytest'].size))[0]
    
    return (trainData, trainLabel, testData, testLabel)

def kernel(data_j, data_i, params):
    """
            计算样本j和样本i的内积
    """
    if params.kernelType == 'linear':
        k = np.dot(data_j, data_i)
    if params.kernelType == 'gaussian':
        diff = data_j - data_i
        diff = np.dot(diff, diff)
        k = np.exp(- diff / (2.0 * np.square(params.sigma)))
    
    return k

def gOfx(data, label, data_i, params):
    """
            计算g(x)=sum(ai * yi * k(xi, x)) + b
    """
    if params.kernelType == 'linear':
        k = np.dot(data, data_i) 
    if params.kernelType == 'gaussian':
        diff = np.subtract(data, data_i)
        diff = np.sum(np.multiply(diff, diff), axis = 1)
        k = np.exp(- diff / (2.0 * np.square(params.sigma)))
    g = np.dot(np.multiply(params.a ,label), k) + params.b
    return g

def init(data, label, params):
    """
            初始化E, a
    """
    params.a = np.zeros((1, len(label)))[0]
    for idx in range(len(params.a)):
        diff = gOfx(data, label, data[idx], params) - label[idx]
        params.e.append(diff)

def updateE(data, label, params):
    """
            更新E
    """
    for idx in range(len(params.a)):
        diff = gOfx(data, label, data[idx], params) - label[idx]
        params.e[idx] = diff

def getMaxE(idx, params):
    """
            获取除去第idx个样本后最大的E
    """
    tmp = params.e[1 : idx] + [0] + params.e[idx+1 :]
    return tmp.index(max(tmp))

def getMinE(idx, params):
    """
            获取除去第idx个样本后最小的E
    """
    tmp = params.e[1:idx] + [params.c * 10] + params.e[idx+1 :]
    return tmp.index(min(tmp))

def judgeConvergence(data, label, params):
    """
            判断是否都已满足KKT条件
    """
    alphay = 0.0    
    for idx in range(len(label)):
        yg = label[idx] * gOfx(data, label, data[idx], params)
        if (params.a[idx] == 0) and (yg < 1 - params.epsilon):
            return False
        elif (params.a[idx] > 0) and (params.a[idx] < params.c) and \
            (yg > 1 + params.epsilon) or (yg < 1 - params.epsilon):
            return False
        elif (params.a[idx] == params.c) and (yg > 1 + params.epsilon):
            return False
        if (params.a[idx] < 0) or (params.a[idx] > params.c):
            return False
        alphay += params.a[idx] * label[idx]
    if abs(alphay) > params.epsilon:
        return False
    return True

def train(data, label, params):
    """
    svm train, use smo solve the quadratic convex problem
    """
    iters = 0
    init(data, label, params)
    updated = True
    while iters < params.maxIter and updated:
        updated = False
        iters += 1
        ##这里直接遍历所有的样本，只要违反了KKT条件就进入迭代
        for i in range(len(params.a)):
            ai = params.a[i]
            ei = params.e[i]
            
            if (label[i] * ei < params.stopTol and ai < params.c) or \
                (label[i] * ei > params.stopTol and ai > 0):
                if ei > 0:
                    j = getMinE(i, params)
                else:
                    j = getMaxE(i, params)
                
                eta = kernel(data[i], data[i], params) + \
                    kernel(data[j], data[j], params) - \
                    2 * kernel(data[i], data[j], params)
                if eta <= 0:
                    continue
                new_aj = params.a[j] + label[j] * \
                    (params.e[i] - params.e[j]) / eta
                L = 0.0
                H = 0.0
                if label[i] == label[j]:
                    L = max(0, params.a[j] + params.a[i] - params.c)
                    H = min(params.c, params.a[j] + params.a[i])
                else:
                    L = max(0, params.a[j] - params.a[i])
                    H = min(params.c, params.c + params.a[j] - params.a[i])
                if new_aj > H:
                    new_aj = H
                if new_aj < L:
                    new_aj = L
                
                ##判断alpha j是否有较大的变化
                if abs(params.a[j] - new_aj) < 0.001:
                    print "j= %d, is not moving enough" % j
                    continue
                new_ai = params.a[i] + label[i] * label[j] * (params.a[j] - new_aj)
                new_b1 = params.b - params.e[i] - label[i] * kernel(data[i], data[i], params) * (new_ai - params.a[i]) \
                    - label[j] * kernel(data[j], data[i], params) * (new_aj - params.a[j])
                new_b2 = params.b - params.e[j] - label[i] * kernel(data[i], data[j], params) * (new_ai - params.a[i]) \
                    - label[j] * kernel(data[j], data[j], params) * (new_aj - params.a[j])
                if new_ai > 0 and new_ai < params.c:
                    new_b = new_b1
                elif new_aj > 0 and new_aj < params.c:
                    new_b = new_b2
                else:
                    new_b = (new_ai + new_aj)/2.0
                    
                params.a[i] = new_ai
                params.a[j] = new_aj
                params.b = new_b
                updateE(data, label, params)
                updated = True
                print "iterate: %d, change pair: i: %d, j: %d" % (iters, i, j)
        res = judgeConvergence(data, label, params)
        if res:
            break

def predictAccuracy(trainData, trainLabel, testData, testLabel, params):
    """
            计算测试集的精度
    """
    pred = np.zeros((1, len(testLabel)))
    for idx in range(len(testLabel)):
        pred[0, idx] = gOfx(trainData, trainLabel, testData[idx], params)
    accuracy = np.mean(np.sign(pred) == testLabel)
    return accuracy

def getSupportVector(params):
    """
            获取支持向量的index
    """
    supportVector = []
    for idx in range(len(params.a)):
        if params.a[idx] > 0 and params.a[idx] < params.c:
            supportVector.append(idx)
    return supportVector
            
def draw(trainData, trainLabel, params):
    """
            特征数是两维时，可以画图来展示
    """
    plt.xlabel(u'x1')
    plt.xlim(0, 100)
    plt.ylabel(u'x2')
    plt.ylim(0, 100)
    plt.title('svm - %s, tolerance %f, c %f' % ('trainData', params.stopTol, params.c))
    for idx in range(len(trainLabel)):
        data = trainData[idx]
        if int(trainLabel[idx]) > 0:
            plt.plot(data[0], data[1], 'or')
        else:
            plt.plot(data[0], data[1], 'xg')
    
    w1 = 0.0
    w2 = 0.0
    for i in range(len(trainLabel)):
        w1 += params.a[i] * trainLabel[i] * trainData[i][0]
        w2 += params.a[i] * trainLabel[i] * trainData[i][1]
    w = - w1 / w2
    
    b = - params.b / w2
    r = 1 / w2
    lp_x1 = [10, 90]
    lp_x2 = []
    lp_x2up = []
    lp_x2down = []
    for x1 in lp_x1:
        lp_x2.append(w * x1 + b)
        lp_x2up.append(w * x1 + b + r)
        lp_x2down.append(w * x1 + b - r)
    plt.plot(lp_x1, lp_x2, 'b')
    plt.plot(lp_x1, lp_x2up, 'b--')
    plt.plot(lp_x1, lp_x2down, 'b--')
    plt.show()
    
    
def main():
####------------------------------------------
#     filePath = '../smo/train.txt'
#     fileType = 'txt'
#     trainData, trainLabel, testData, testLabel = loadData(filePath, fileType)
#     params = Params()
#     params.setKernel('linear')
    
####------------------------------------------
    filePath = '../smo/ionosphere_1.mat'
    fileType = 'mat'
    trainData, trainLabel, testData, testLabel = loadData(filePath, fileType)
    params = Params()
#     params.setKernel("gaussian", 0.5) #acc=0.902857  100步截断
    params.setKernel("gaussian", 1) ## acc = 0.942857  18步收敛
    params.setMaxIter(500) ##在230步左右收敛，精度为0.88,sv 太多
    params.setEpsilon(0.01)

####------------------------------------------

    train(trainData, trainLabel, params)
    print 'a= ', params.a
    print 'b= ', params.b
         
    supportVector = getSupportVector(params)
    print 'support vector = ', supportVector
    
#### 两维特征可以在样本空间中画出图来展示划分效果
#     draw(trainData, trainLabel, params)

#### 有测试集可以计算出预测精度
    print('test set accuracy: %f' % predictAccuracy(trainData, trainLabel, testData, testLabel, params))

if __name__ == '__main__':
    main()
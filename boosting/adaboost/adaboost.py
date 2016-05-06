# coding: utf-8
import numpy as np

class Params:
    def __init__(self):
        self.w = []
        self.alpha = []
        self.threshold = []
        self.g = []
    
    def setW(self, w):
        self.w = w

def getThreshold(x, y, params):
    """获取当前权值分布下的误差率最小的阈值"""
    threshold = np.zeros(len(x) + 1)
    for idx in range(len(x)):
        if idx > 0:
            threshold[idx] = np.mean(x[idx-1 : idx+1])
        else:
            threshold[idx] = x[idx] - 0.5 ## 延拓0.5
    threshold[-1] = x[-1] + 0.5    
    predict = np.ones(len(y)) * -1
    minError = 1000
    minThreshold = 0
    ming = 1 ## 表示小于阈值时取1 否则取-1
    for idx in range(len(threshold)):
#         predict = np.sign(x < threshold[idx], dtype = int)
        predict = map(lambda x1: 1 if x1 == True else -1, x < threshold[idx])
        error = np.sum(params.w[predict != y]) ##这里用样本权重来衡量误差率
        if error < minError:
            minThreshold = idx
            minError = error
#         predict = np.ones(len(y)) * -1 
    for idx in range(len(threshold)):
#         predict = np.sign(x > threshold[idx], dtype = int)
        predict = map(lambda x1: 1 if x1 == True else -1, x > threshold[idx])
        error = np.sum(params.w[predict != y])
        if error < minError:
            minThreshold = idx
            minError = error
            ming = -1
#         predict = np.ones(len(y)) * -1
    params.threshold.append(threshold[minThreshold])
    params.g.append(ming)
    return minError

def getGofx(idx, x, params):
    """获取第idx个分类器的预测值"""
    if params.g[idx] == 1:
        G = map(lambda x1: 1 if x1 == True else -1, x < params.threshold[idx])
    else:
        G = map(lambda x1: 1 if x1 == True else -1, x > params.threshold[idx])
    return np.array(G)

def updateWeight(x, y, params):
    """更新样本权重"""
    G = getGofx(-1, x, params)
    normer = np.exp(- params.alpha[-1] * y * G)
    Z = np.dot(params.w, normer)
    params.w = params.w * normer / Z

def getPredict(x, params):
    """获取当前分类器的预测"""
    predict = np.zeros(len(x))
    for idx in range(len(params.alpha)):
        G = getGofx(idx, x, params)
        predict += G * params.alpha[idx]
    predict = np.sign(predict)
    return predict
    
def adaboost(x, y, params):
    """adaboost训练过程"""
    errorNum = len(y)
    while(errorNum > 0):
        errorRate = getThreshold(x, y, params)
        
        alpha = 0.5 * np.log((1 - errorRate)/errorRate)
        params.alpha.append(alpha)
        
        updateWeight(x, y, params)
        predict = getPredict(x, params)
        error = predict != y
        errorNum = len(y[error])
        print "alpha:", params.alpha
        print "threshold:", params.threshold
        print "errorNum:", errorNum

def main():
    x = np.array([0,1,2,3,4,5,6,7,8,9])
    y = np.array([1,1,1,-1,-1,-1,1,1,1,-1])
    params = Params()
    params.setW(np.ones(len(y)) * (1.0 / len(y)))
    adaboost(x, y, params)
    
    predict = getPredict(x, params)
    print "y = ", y
    print "predict = ", predict
    
if __name__ == '__main__':
    main()
        
        
        
        
        
        
        
        
        
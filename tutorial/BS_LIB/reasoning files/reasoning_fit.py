import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq

def logExpFunc():
    '''
    【指数+对数】混合函数拟合
    '''
    x=np.linspace(1,2,15)
    y=[21.5,23.1,25.9,30,32.6,38,41.9,47.2,55,61,69,80,90,105,117.6]
    popt,pcov=curve_fit(lambda t,a,b,c,d,e: a+b*np.log(c*t)+d*np.exp(e*t),x,y)
    print('popt: ',popt)
    plt.figure()
    a,b,c,d,e=popt
    y_pre = a+b*np.log(c*x)+d*np.exp(e*x)
    plt.plot(x, y, 'ko', label="Original Data")
    plt.plot(x, y_pre, 'r-', label="Fitting Curve")
    plt.legend()
    plt.show()

'''
def Fun(x,a1,a2,a3):                   # 定义拟合函数形式
    return a1*x**2+a2*x+a3
def error (p,x,y): # 拟合残差
 return Fun(p,x)-y
def main():
    x = np.linspace(-10,10,100)       # 创建时间序列
    a1,a2,a3 = [-2,5,10]              # 原始数据的参数
    noise = np.random.randn(len(x))   # 创建随机噪声
    y = Fun(x,a1,a2,a3)+noise*2       # 加上噪声的序列
    para,pcov=curve_fit(Fun,x,y)
    y_fitted = Fun(x,para[0],para[1],para[2]) # 画出拟合后的曲线
 
    plt.figure
    plt.plot(x,y,'r', label = 'Original curve')
    plt.plot(x,y_fitted,'-b', label ='Fitted curve')
    plt.legend()
    plt.show()
    print (para)
'''

def Fun(p,x):                        # 定义拟合函数形式
    a1,a2,a3 = p
    return a1*x**2+a2*x+a3
def error (p,x,y):                    # 拟合残差
    return Fun(p,x)-y 
def main():
    x = np.linspace(-10,10,100)  # 创建时间序列
    p_value = [-2,5,10] # 原始数据的参数
    noise = np.random.randn(len(x))  # 创建随机噪声
    y = Fun(p_value,x)+noise*2 # 加上噪声的序列
    p0 = [0.1,-0.01,100] # 拟合的初始参数设置
    para =leastsq(error, p0, args=(x,y)) # 进行拟合
    y_fitted = Fun (para[0],x) # 画出拟合后的曲线
 
    plt.figure
    plt.plot(x,y,'r', label = 'Original curve')
    plt.plot(x,y_fitted,'-b', label ='Fitted curve')
    plt.legend()
    plt.show()
    print (para[0])


 
if __name__=='__main__':
   logExpFunc()
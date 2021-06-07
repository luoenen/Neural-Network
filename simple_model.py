'''
@ project: Neural-Network
@ file: simple_model
@ user: 罗申申
@ email: luoshenshen@buaa.edu.cn
@ tool: PyCharm
@ time: 2021/6/7 17:00
'''
import numpy as np
from matplotlib import pyplot as plt

x_date = [1.0, 2.0, 3.0, 4.0]
y_date = [3.0, 6.0, 9.0, 12.0]

x_test = 5.0
y_value = 15.0

def forward(x):
    return w * x

def get_loss(x,y):
    pred_y = forward(x)
    return (pred_y - y) * (pred_y - y)

w_list = []
loss_list = []
pred_y_loss = []

for w in np.arange(1.0,5.0,0.01):
    loss_sum = 0
    for x,y in zip(x_date,y_date):

        loss = get_loss(x,y)
        loss_sum+=loss
    w_list.append(w)
    loss_list.append(loss_sum/len(x_date))
    pred_y_loss.append(w * x_test-y_value)
    print('w = ', w, '平均loss = ', loss_sum/len(x_date))
    print("测试值:",w * x_test,'真实值:',y_value,"差距:",w * x_test-y_value)
plt.plot(w_list,loss_list)
plt.xlabel('w')
plt.ylabel('loss')
plt.show()

plt.plot(w_list,pred_y_loss)
plt.xlabel('w')
plt.ylabel('pred_y_loss')
plt.show()
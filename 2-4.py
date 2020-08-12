import tensorflow as tf
import numpy as np
#真实数据
x_data=np.random.rand(100)
y_data=x_data*0.1+0.2
#构造模型
k=tf.Variable(0.0)
b=tf.Variable(0.0)
y=k*x_data+b
#计算损失
loss=tf.reduce_mean(tf.square(y_data-y))#取所有值的平均值
#设置优化器，梯度下降
optimizer = tf.train.GradientDescentOptimizer(0.2)
#最小化损失
train = optimizer.minimize(loss)

#初始化变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 == 0:
            print(step,sess.run([k,b]))

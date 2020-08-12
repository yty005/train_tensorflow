import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

#设置两个placeholder
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

#设置中间层
Weight_L1=tf.Variable(tf.random_normal([1,10]))
baise_L1=tf.Variable(tf.zeros([1,10]))
X_W_B_L1=tf.matmul(x,Weight_L1)+baise_L1
L1=tf.nn.tanh(X_W_B_L1)

#设置输出层
Weight_out=tf.Variable(tf.random_normal([10,1]))
baise_out=tf.Variable(tf.zeros([1,1]))
L1_W_B_out=tf.matmul(L1,Weight_out)+baise_out
out=tf.nn.tanh(L1_W_B_out)

#设置损失值
loss=tf.reduce_mean(np.square(y-out))
#设置优化器
optimizer=tf.train.GradientDescentOptimizer(0.05)
#设置最小化
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(2000):
        sess.run(train,feed_dict={x:x_data,y:y_data})
    prediction=sess.run(out,feed_dict={x:x_data})
    #开始画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction,'r-',lw=3)
    plt.show()
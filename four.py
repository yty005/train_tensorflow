import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#共三层，一层输入层，一层输出层，一层隐藏层
#添加神经网络层
def add_layer(input,in_size,out_size,activation_function=None):
    Weight=tf.Variable(tf.random_normal([in_size,out_size]))#设置随机权值
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)#设置偏差 数组里的每个元素增加0.1
    Wx_plus_b=tf.matmul(input,Weight)+biases # y=xw+b #设置输出
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs
#训练数据
x_data =np.linspace(-1,1,300)[:,np.newaxis]#300条数据 每条数据一个输入x_data,输出一个y_data
noise= np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data) - 0.5 + noise
#print(x_data.shape)#300*1
#print(y_data.shape)#300*1
xs=tf.placeholder(tf.float32,[None,1])#None给多少个例子都可以 1表示只有1个输入
ys=tf.placeholder(tf.float32,[None,1])#同上

l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction=add_layer(l1,10,1,activation_function=None)
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))#0意味着将矩阵压缩为一行 ，1意味着将矩阵压缩为1列
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)#0.1表示学习率，表示到达最优解的快慢，minmize()计算梯度，更新参数

init =tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data.y_data)
plt.ion()
plt.show()

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})#使用传入值则必须使用feed_dict字典
    if i% 50 ==0:
        #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value=sess.run(prediction,feed_dict={xs:x_data})
        lines=ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.1)





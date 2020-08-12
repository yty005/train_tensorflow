import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
minst=input_data.read_data_sets("D:\minst",one_hot=True)

#设置批次
batch_size=100
n_batch=minst.train.num_examples // batch_size

#设置两个占位符
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
keep_drop=tf.placeholder(tf.float32)

#构造神经网络
Weight_L1 = tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
baise_L1 = tf.Variable(tf.zeros([2000])+0.1)
Wx_plus_b_L1 = tf.matmul(x,Weight_L1)+baise_L1
L1=tf.nn.dropout(tf.nn.tanh(Wx_plus_b_L1),keep_prob=keep_drop)

Weight_L2 = tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
baise_L2 = tf.Variable(tf.zeros([2000])+0.1)
Wx_plus_b_L2 = tf.matmul(L1,Weight_L2)+baise_L2
L2=tf.nn.dropout(tf.nn.tanh(Wx_plus_b_L2),keep_prob=keep_drop)

Weight_L3 = tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
baise_L3 = tf.Variable(tf.zeros([1000])+0.1)
Wx_plus_b_L3 = tf.matmul(L2,Weight_L3)+baise_L3
L3=tf.nn.dropout(tf.nn.tanh(Wx_plus_b_L3),keep_drop)

Weight_out = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
baise_out = tf.Variable(tf.zeros([1,10])+0.1)
Wx_plus_b_out = tf.matmul(L3,Weight_out)+baise_out
prediction = tf.nn.softmax(Wx_plus_b_out)

#设置损失值
#loss = tf.reduce_mean(tf.square(y-prediction))
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#设置优化器
optimizer = tf.train.GradientDescentOptimizer(0.15)
train_step=optimizer.minimize(loss)

#设置准确率
correct = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax函数，取每一行最大值位置

accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))#cast格式转换

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for step in range(n_batch):
            batch_xs,batch_ys=minst.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_drop:0.7})
        test_acc = sess.run(accuracy,feed_dict={x:minst.test.images,y:minst.test.labels,keep_drop:1.0})
        train_acc = sess.run(accuracy,feed_dict={x:minst.train.images,y:minst.train.labels,keep_drop:1.0})
        print("Iter " + str(epoch) + ",train_Accuracy " + str(train_acc) + ",test_Accuracy "+str(test_acc))



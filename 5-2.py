import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
minst=input_data.read_data_sets("D:\minst",one_hot=True)

#设置批次
batch_size=100
n_batch=minst.train.num_examples // batch_size

with tf.name_scope('input'):
    #设置两个占位符
    x = tf.placeholder(tf.float32,[None,784],name='x_input')
    y = tf.placeholder(tf.float32,[None,10],name='y_input')

with tf.name_scope('layer'):
    #构造神经网络
    with tf.name_scope('weight'):
        Weight_out = tf.Variable(tf.zeros([784,10]))
    with tf.name_scope('baise'):
        baise = tf.Variable(tf.zeros([1,10]))
    with tf.name_scope('wx_plus_b'):
        Wx_plus_b_out = tf.matmul(x,Weight_out)+baise
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(Wx_plus_b_out)

#设置损失值
#loss = tf.reduce_mean(tf.square(y-prediction))
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#设置优化器
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step=optimizer.minimize(loss)

#设置准确率
correct = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax函数，取每一行最大值位置

accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))#cast格式转换

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(21):
        for step in range(n_batch):
            batch_xs,batch_ys=minst.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict={x:minst.test.images,y:minst.test.labels})
        print("Iter " + str(epoch) + " Accuracy " + str(acc))



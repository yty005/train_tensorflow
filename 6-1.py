import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

minst = input_data.read_data_sets('D:\minst',one_hot=True)
batch_size=100
n_batch=minst.train.num_examples // batch_size

#初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#设置偏置量
def baise_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#卷积层
def conv(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')#conv2d  [filter_height,filter_width,in_channels,out_channels]  stride[1]横向步长 stride[2]纵向步长

#池化层
def pooling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#定义两个占位符
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#将x转为4d向量
x_image = tf.reshape(x,[-1,28,28,1])

#设置权值偏置值
w_convL1=weight_variable([5,5,1,32])
b_convL1=baise_variable([32])

L1 = tf.nn.relu(conv(x_image,w_convL1)+b_convL1)
L1_pooling = pooling(L1)

w_convL2 = weight_variable([5,5,32,64])
b_convL2 = baise_variable([64])

L2 = tf.nn.relu(conv(L1_pooling,w_convL2) + b_convL2)
L2_pooling = pooling(L2)

#全连接层
w_fc1 = weight_variable([7*7*64,1024])
b_fc1 = baise_variable([1024])

L2_pooling_flat = tf.reshape(L2_pooling,[-1,7*7*64])
fc1 = tf.nn.relu(tf.matmul(L2_pooling_flat,w_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)
fc1_drop = tf.nn.dropout(fc1,keep_prob)

#全连接层2
w_fc2 = weight_variable([1024,10])
b_fc2 = baise_variable([10])

prediction = tf.nn.softmax(tf.matmul(fc1_drop,w_fc2)+b_fc2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys=minst.train.next_batch(batch_size)
            #print(batch_xs.shape,batch_ys.shape)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
        acc = sess.run(accuracy,feed_dict={x:minst.test.images,y:minst.test.labels,keep_prob:1.0})
        print('epoch:' + str(epoch) + ',Accuracy:' + str(acc))

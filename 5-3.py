import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
minst=input_data.read_data_sets("D:\minst",one_hot=True)

#设置批次
batch_size=100
n_batch=minst.train.num_examples // batch_size

#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)

with tf.name_scope('input'):
    #设置两个占位符
    x = tf.placeholder(tf.float32,[None,784],name='x_input')
    y = tf.placeholder(tf.float32,[None,10],name='y_input')

with tf.name_scope('layer'):
    #构造神经网络
    with tf.name_scope('weight'):
        Weight_out = tf.Variable(tf.zeros([784,10]))
        variable_summaries(Weight_out)
    with tf.name_scope('baise'):
        baise = tf.Variable(tf.zeros([1,10]))
        variable_summaries(baise)
    with tf.name_scope('wx_plus_b'):
        Wx_plus_b_out = tf.matmul(x,Weight_out)+baise
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(Wx_plus_b_out)

with tf.name_scope('loss'):
    #设置损失值
    #loss = tf.reduce_mean(tf.square(y-prediction))
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    #设置优化器
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train_step=optimizer.minimize(loss)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct'):
        #设置准确率
        correct = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax函数，取每一行最大值位置
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))#cast格式转换
        tf.summary.scalar('sccuracy',accuracy)

#合并所有的summary
merged=tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(51):
        for step in range(n_batch):
            batch_xs,batch_ys=minst.train.next_batch(batch_size)
            summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})
        writer.add_summary(summary,epoch)
        acc = sess.run(accuracy,feed_dict={x:minst.test.images,y:minst.test.labels})
        print("Iter " + str(epoch) + " Accuracy " + str(acc))



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

#载入数据集
minst = input_data.read_data_sets('D:\minst',one_hot=True)
#运行次数
max_steps=1001
#图片数量
image_num=3000
#文件路径
DIR='D:/trainone/'

#定义会话
sess=tf.Session()

#载入图片
embedding = tf.Variable(tf.stack(minst.test.images[:image_num]),trainable=False,name='embedding')

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

#显示图片
with tf.name_scope('input_reshape'):
    image_shaped_input=tf.reshape(x,[-1,28,28,1])
    tf.summary.image('input',image_shaped_input,10)

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

if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')
with open(DIR + 'projector/projector/metadata.tsv','w') as f:
    labels = sess.run(tf.argmax(minst.test.labels[:],1))
    for i in range(image_num):
        f.write(str(labels[i])+'\n')

#合并所有的summary
merged=tf.summary.merge_all()

init = tf.global_variables_initializer()

projector_writer=tf.summary.FileWriter(DIR + 'projector/projector',sess.graph)
saver=tf.train.Saver()
config=projector.ProjectorConfig()
embed=config.embeddings.add()
embed.tensor_name=embedding.name
embed.metadata_path=DIR + 'projector/projector/metadata.tsv'
embed.sprite.image_path=DIR + 'projector/data/minst_10k_sprite.ong'
embed.sprite.single_image_dim.extend([28,28])
projector.visualize_embeddings(projector_writer,config)

sess.run(init)
for i in range(max_steps):
    batch_xs,batch_ys=minst.train.next_batch(100)
    run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata=tf.RunMetadata()
    summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys},options=run_options,run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata,'step%03d'%i)
    projector_writer.add_summary(summary,i)
    if i%100 ==0:
        acc = sess.run(accuracy,feed_dict={x:minst.test.images,y:minst.test.labels})
        print("Iter " + str(i) + " Accuracy " + str(acc))
saver.save(sess,DIR + 'projector/projector/a_model.ckpt',global_step=max_steps)
projector_writer.close()
sess.close()

#传入值
import tensorflow as tf
# input1=tf.placeholder(tf.float32)
# input2=tf.placeholder(tf.float32)
#
# output=tf.multiply(input1,input2)
# with tf.Session() as sess:
#     print(sess.run(output,feed_dict={input1:[12.],input2:[3.]}))
a=tf.Variable(tf.zeros([1,5])+0.1)
sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init)
print(sess.run(a))
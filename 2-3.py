import tensorflow as tf
# #fetch
# input1=tf.constant([3.0])
# input2=tf.constant([2.0])
# input3=tf.constant([5.0])
#
# add=tf.add(input2,input3)
# mul=tf.multiply(add,input1)
#
# with tf.Session() as sess:
#     result=sess.run([mul,add])
#     print(result)
#feed
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)
output=tf.multiply(input1,input2)

with tf.Session() as sess:
    result=sess.run(output,feed_dict={input1:[5.0],input2:[2.0]})
    print(result)
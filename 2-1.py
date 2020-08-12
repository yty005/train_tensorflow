import tensorflow as tf

a=tf.constant([[1,2]])
b=tf.constant([[2],[1]])
c=tf.matmul(a,b)

sess=tf.Session()
result=sess.run(c)
print(result)
sess.close()
#session会话控制
import tensorflow as tf
matrix1=tf.constant([[3,3]])
matrix2=tf.constant([[2],[1]])
product=tf.matmul(matrix1,matrix2)

# #method 1  #ctrl+/多行注释  取消注释也是Ctrl+/
# sess=tf.Session()
# result=sess.run(product)
# print(result)

#method 2
with tf.Session() as sess:
    result2=sess.run(product)
    print(result2)
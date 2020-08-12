import tensorflow as tf

# a=tf.Variable([1,2])
# b=tf.constant([2,3])
# #减法
# sub=tf.subtract(b,a)
# #加法
# add=tf.add(sub,a)
#
# init=tf.global_variables_initializer()
# sess=tf.Session()
# sess.run(init)
# result_sub=sess.run(sub)
# result_add=sess.run(add)
# print(result_sub)
# print(result_add)

state=tf.Variable(0,name='counter')
new_state=tf.add(state,1)
update=tf.assign(state,new_state)

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        print(sess.run(update))


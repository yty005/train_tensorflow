import tensorflow as tf

#变量
state =tf.Variable(0,name='counter')
#print(state.name)
one =tf.constant(1)

new_value=tf.add(state,one)
updata=tf.assign(state,new_value)

init=tf.initialize_all_variables()#如果定义里变量则必须使用

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(updata)
        print(sess.run(state))


# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:06:37 2019

@author: Suyash
"""

import tensorflow as tf
const= tf.constant(value=[[1.0,2,0]],dtype=tf.float32,name='constant_1')
const_2= tf.constant(value=[[2.0,2,0]],dtype=tf.float32,name='constant_2')

session = tf.Session()
print(session.run(fetches=[const,const_2]))

var_1= tf.Variable(initial_value=[0,1],trainable=True,name='Variable_1')
init = tf.global_variables_initializer()
session.run(init)
print(session.run(fetches=[var_1]))
var_2 = var_1.assign(value=[1,5])
print(session.run(fetches=[var_2]))

placeholder_1=tf.placeholder(dtype= tf.float32,name='PlaceHolder1')
placeholder_2=tf.placeholder(dtype= tf.float32,name='PlaceHolder2')

print(placeholder_1)
session.run(fetches=placeholder_1,feed_dict={placeholder_1:[[2.7,3.2],[2.2,3.1]],placeholder_2:[[2.9,3.0],[2.12,3.33]]})


const_3= tf.constant(value=[[1.0]],dtype=tf.float32,name='constant_1')
const_4= tf.constant(value=[[2.0]],dtype=tf.float32,name='constant_2')
results = const_3 + const_4
session.run(fetches=results)
result=tf.add(x=const_3,y=const_4)
session.run(fetches=result)
placehold_1=tf.placeholder(dtype=tf.float32)
res=tf.add(x=placehold_1,y=const_4)
session.run(fetches=res,feed_dict={placehold_1:[[7.3]]}) 
 # y= WX + b
W=tf.constant(value=[2.0])
b=tf.constant(value=[1.0])

x=tf.placeholder(dtype=tf.float32)

y=W * x +b

print(session.run(fetches=y,feed_dict={x:[2.0]}))

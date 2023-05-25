import tensorflow as tf
import numpy as np

#detaset

x=np.array([-1.0,0.0,1.0,2.0,3.0,4.0],dtype=float)
y=np.array([-3.0,-1.0,0.0,1.0,3.0,4.0],dtype=float)


#simple neural network
from tensorflow import  keras
model=tf.keras.Sequential(tf.keras.layers.Dense(units=1,input_shape=[1]))

#compiling
model.compile(optimizer="sgd",loss="mean_squared_error")

#model traning
model.fit(x,y,epochs=500)

#prediction
print(model.predict([5.0]))
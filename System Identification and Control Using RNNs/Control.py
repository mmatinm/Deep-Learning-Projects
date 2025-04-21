
''' #if needed 
pip install control  
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
import tensorflow.keras as keras
from tensorflow.keras import layers, models, Sequential
import control
from sklearn.preprocessing import StandardScaler
from scipy import signal
import tensorflow as tf
from keras.layers import Dense , Dropout ,SimpleRNN ,LSTM
from keras.optimizers import SGD
from keras.optimizers import Adam , RMSprop
from scipy import integrate
'''
# if you are using google colab 
from google.colab import drive
drive.mount('/content/new_drive')
'''
############### generating data
m = 24
c = 110
k = 2500

tts = np.linspace(0, 10, 500)
forcets = 4*(signal.square(2 * np.pi * 0.2 * tts))

systemts = control.TransferFunction([1], [m, c, k])
tts, positionts = control.forced_response(systemts, T=tts, U=forcets)
velocityts = np.gradient(positionts, tts)
#accelerationts = np.gradient(velocityts, tts)

scaler_Xts = StandardScaler()
scaler_yts = StandardScaler()

forcets_normalized = scaler_Xts.fit_transform(forcets.reshape(-1, 1))
positionts_normalized = scaler_yts.fit_transform(positionts.reshape(-1, 1))

mean_force = float(scaler_Xts.mean_)
std_dev_force = float(scaler_Xts.scale_)
mean_pos = float(scaler_yts.mean_)
std_dev_pos = float(scaler_yts.scale_)

'''
load the trained model from the previous part. 
this model represent the dynamic system which we want to control its position.
'''
### adjust the path
trainedmodel = tf.keras.models.load_model('/content/new_drive/MyDrive/trained_SRNN_model.keras') 

# applying proportional controller to the system
time_steps = 500
setpoint = 0.001
Kp = 4000 #4300 #4750
#Ke = 200  #uncomment this if you want to build a PD controller

F = np.zeros((1, time_steps, 1))
f_in = np.zeros((1, time_steps, 1))
f = np.zeros((time_steps, 1))
output = np.zeros((time_steps, 1))

for t in range(2, time_steps):
    f[t] =  (Kp*(setpoint - float(output[t-1])) - mean_force)/std_dev_force  #Ke*(float(output[t-2])-float(output[t-1]))/0.02   #uncomment this and add it to the equation if you want to build a PD controller
    f_in[0,:,0] = f.flatten()  #[t-1,0]
    F[0,t,0] = f_in[0,t,0]
    pred = trainedmodel.predict(f_in)# .numpy()
    p = pred.reshape((pred.shape[1], pred.shape[0]))
    output[t] = float(p[t,0])*std_dev_pos + mean_pos
    #print(t)

time = np.arange(time_steps)*0.02
pe1 = tf.reshape(pred , (pred.shape[0], pred.shape[1]))
pe = pe1.numpy()
# Plot Position vs time and setpoint
plt.subplot(2, 1, 1)
plt.plot(time, output, label='Position')
plt.plot(time, setpoint * np.ones_like(time), label='Setpoint', linestyle='--')
#plt.xlabel('Time')
plt.ylabel('Position')
plt.title('P Controller Performance ')


# Plot f vs time
plt.subplot(2, 1, 2)
plt.plot(time, F[0, :, 0], label='f')
plt.xlabel('Time')
plt.ylabel('Control Input scaled ')

plt.show()

######### preprocessing data to the desierd shape

X1_train = ((setpoint - output) - mean_pos)/std_dev_pos
y1_train = f
X1_train = X1_train.reshape((1,X1_train.shape[0],X1_train.shape[1]))
y1_train = y1_train.reshape((1,y1_train.shape[0],y1_train.shape[1]))
print(X1_train.shape)
print(y1_train.shape)

# building a neural network as controller and train it so it can learn the proportional controller behaviour
ctrl = Sequential()
ctrl.add(LSTM(6, input_shape=(X1_train.shape[1],X1_train.shape[2]),return_sequences=True , name = 'l1'))  #use_bias=False
ctrl.add(Dense(1, activation='linear' , name = 'l2')) #use_bias=False
ctrl.compile(optimizer=RMSprop(learning_rate=0.001 , rho=0.9), loss='mean_squared_error')

ctrl.fit(X1_train, y1_train, epochs=400, batch_size=1,verbose=1)
'''
# check the performance of ctrl model in learning the behaviour of proportional controller
Pred = ctrl.predict(X1_train)
plt.scatter(time, Pred[0,:,0], label='Control input', c='blue', s=4)
#plt.plot(time, Pred[0,:,0], label='Zero', c='green')
plt.plot(time, F[0, :, 0], label='f Proportional', c='orange')
plt.legend()
plt.title('Initial NNC Train Performance')
plt.xlabel('Time (s)',size =12)
plt.ylabel('Control Input scaled ',size =12)

plt.show()

'''

############ training
'''
first we built a model consist of controller and plant. second we freezed the plant so its weights does'nt change
in training. third we used the control systems approch to feed in the error and checking the output. 
in the end we are trying to 
'''
for layer in trainedmodel.layers:
    layer.trainable = False

ctrlxplant = Sequential()
ctrlxplant.add(ctrl)
ctrlxplant.add(trainedmodel)

time_steps = 500
setpoint = (0.001 - mean_pos)/std_dev_pos
E = np.zeros((1, time_steps, 1))
e_in = np.zeros((1, time_steps, 1))
e = np.zeros((time_steps, 1))          # error signal
outputn = np.zeros((time_steps, 1))
s = np.ones_like(outputn)*setpoint     # setpoint
s = s.reshape((1,s.shape[0],s.shape[1]))
ctrlxplant.compile(optimizer=RMSprop(learning_rate=0.0002 , rho=0.95), loss='mean_squared_error')

def smooth_error(e, alpha=0.9):
    smoothed = np.zeros_like(e)
    smoothed[0] = e[0]
    for i in range(1, len(e)):
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * e[i]
    return smoothed

for t in range(0, time_steps-1):
    e[t] = (setpoint - float(outputn[t]))
    #e = smooth_error(e)
    e_in[0,:,0] = e.flatten()
    E[0,t,0] = e_in[0,t,0]
    n = int(6 - t/100)

    #if t % 5 == 0:
    ctrlxplant.fit(e_in, s, epochs=n , batch_size=1,verbose=1)

    prede = ctrlxplant.predict(e_in)
    p = prede.reshape((prede.shape[1], prede.shape[0]))
    outputn[t+1] = float(p[t,0])

  #ctrlxplant.fit(e_in, s, epochs=5 , batch_size=1,verbose=1)

outputn = outputn*std_dev_pos + mean_pos
setpoint = setpoint*std_dev_pos + mean_pos

time = np.arange(time_steps)*0.02

#save the trained model
ctrlxplant.save('/content/new_drive/MyDrive/ctrlxplantfinal.keras')
# load the traindmodel ( controller and plant together ) 
## please adjust the path as you need
trainedfinal = tf.keras.models.load_model('/content/new_drive/MyDrive/ctrlxplantfinal.keras') 

# saving the controller network seperately
ctrl_new = Sequential()
ctrl_new.add(LSTM(6, input_shape=(X1_train.shape[1], X1_train.shape[2]), return_sequences=True, name='l1'))
ctrl_new.add(Dense(1, activation='linear', name='l2'))
ctrl_new.set_weights(ctrlxplant.layers[0].get_weights())
ctrl_new.save('/content/new_drive/MyDrive/ctrl_final.keras')

# test the final model and plot the position and force (generated by controller )
time_steps = 500
setpoint1 = (0.001 - mean_pos)/std_dev_pos
E1 = np.zeros((1, time_steps, 1))
e1_in = np.zeros((1, time_steps, 1))
e1 = np.zeros((time_steps, 1))
outputn1 = np.zeros((time_steps, 1))
outputn2 = np.zeros((time_steps, 1))
s1 = np.ones_like(outputn1)*setpoint1
s1 = s1.reshape((1,s1.shape[0],s1.shape[1]))
#ctrlxplant.compile(optimizer=RMSprop(learning_rate=0.00005 , rho=0.9), loss='mean_squared_error')

for t in range(0, time_steps-1):
    e1[t] = (setpoint1 - float(outputn1[t]))
    e1_in[0,:,0] = e1.flatten()
    E1[0,t,0] = e1_in[0,t,0]
    #n = int(30 - time_steps/20)
    #ctrlxplant.fit(e_in, s, epochs=3, batch_size=1,verbose=1)
    prede1 = ctrlxplant.predict(e1_in)  
    p = prede1.reshape((prede1.shape[1], prede1.shape[0]))
    outputn1[t+1] = float(p[t,0])

    prede2 = ctrl_new.predict(e1_in)   
    p = prede2.reshape((prede2.shape[1], prede2.shape[0]))
    outputn2[t+1] = float(p[t,0])


outputn1 = outputn1*std_dev_pos + mean_pos
setpoint1 = setpoint1*std_dev_pos + mean_pos

time = np.arange(time_steps)*0.02
plt.plot(time, outputn1, label='Position')
plt.plot(time, setpoint1 * np.ones_like(time), label='Setpoint', linestyle='--')
#plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Neural Net Controller Performance')
plt.legend()
plt.show()

plt.plot(time, outputn2, label='Force')
plt.ylabel('Force')
plt.title('Controller generated Force')
plt.show()

### evaluating the performance of Neural Net Controller (NNC)
error = setpoint1 - outputn1.flatten()
dt = 0.02
# IAE: Integral of Absolute Error
IAE = np.sum(np.abs(error)) * dt

# ISE: Integral of Squared Error
ISE = np.sum(error**2) * dt

# RMSE: Root Mean Square Error
RMSE = np.sqrt(np.mean(error**2))
relative_IAE = IAE / abs(setpoint1)
relative_ISE = ISE / (setpoint1**2) 
relative_RMSE = RMSE / abs(setpoint1)

print(f"IAE: {IAE:.6f}")
print(f"ISE: {ISE:.6f}")
print(f"RMSE: {RMSE:.6f}")

print(f"Relative IAE: {relative_IAE:.4f}")
print(f"Relative ISE: {relative_ISE:.4f}")
print(f"Relative RMSE: {relative_RMSE:.4f}")


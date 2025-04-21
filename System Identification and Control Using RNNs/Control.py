
''' 
# If the 'control' package is not installed, run: 
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
# If using Google Colab, uncomment and mount Google Drive:
from google.colab import drive
drive.mount('/content/new_drive')
'''
############### Generating synthetic data for the mass-spring-damper system
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
Load the pre-trained system identification model (frozen weights)
This model predicts system position given the scaled force input
'''
### adjust the path
trainedmodel = tf.keras.models.load_model('/content/new_drive/MyDrive/trained_SRNN_model.keras') 

# Apply a proportional controller to the system
time_steps = 500
setpoint = 0.001
Kp = 4000 #4300 #4750
#Ke = 200  #uncomment this if you want to build a PD controller

F = np.zeros((1, time_steps, 1))
f_in = np.zeros((1, time_steps, 1))
f = np.zeros((time_steps, 1))
output = np.zeros((time_steps, 1))

for t in range(2, time_steps):
    # Uncomment the next line and add to the equation for PD control:
    # deriv_term = Ke * (output[t-2] - output[t-1]) / dt
    f[t] =  (Kp*(setpoint - float(output[t-1])) - mean_force)/std_dev_force 
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


# Plot control input (normalized)
plt.subplot(2, 1, 2)
plt.plot(time, F[0, :, 0], label='f')
plt.xlabel('Time')
plt.ylabel('Control Input scaled ')

plt.show()

######### preprocessing data to the desierd shape

X1_train = ((setpoint - output) - mean_pos)/std_dev_pos
X1_train = X1_train.reshape((1,X1_train.shape[0],X1_train.shape[1]))
y1_train = f.reshape((1,y1_train.shape[0],y1_train.shape[1]))
print(X1_train.shape)
print(y1_train.shape)

# Build a simple RNN-based controller model
ctrl = Sequential()
ctrl.add(LSTM(6, input_shape=(X1_train.shape[1],X1_train.shape[2]),return_sequences=True , name = 'l1'))  #use_bias=False
ctrl.add(Dense(1, activation='linear' , name = 'l2')) #use_bias=False
ctrl.compile(optimizer=RMSprop(learning_rate=0.001 , rho=0.9), loss='mean_squared_error')
# Train controller to mimic proportional controller behavior
ctrl.fit(X1_train, y1_train, epochs=400, batch_size=1,verbose=1)

'''
# The following block evaluates initial controller performance (optional)
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

############### Supervised training of combined controller-plant model
'''
Note: framing this control problem as standard supervised learning is unconventional.
Our goal is to drive the system’s position to a fixed setpoint, but we only observe
the error (setpoint minus the current position) at each time step.

To make it tractable with backpropagation, we “stack” the controller and plant
into one end‑to‑end model (“ctrlxplant”). The frozen “trainedmodel” subnetwork
is the system‑identification neural net you generated earlier
(see System Identification.py → trained_SRNN_model.keras).

During training, at each time step:
 1. Compute the error e(t) = setpoint – output(t).
 2. Accumulate this error history into an input sequence.
 3. Use that sequence as input and the desired (normalized) setpoint as target.

We then fit the combined model step‑by‑step.  The number of epochs per step (“n”)
and other hyperparameters were tuned manually.  We found that gradually decreasing
n over time (to counteract stronger transients early on) yielded much better
convergence than using a constant epoch count.
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

for t in range(0, time_steps-1):
    e[t] = (setpoint - float(outputn[t]))
    e_in[0,:,0] = e.flatten()
    E[0,t,0] = e_in[0,t,0]
    n = int(6 - t/100)
    ctrlxplant.fit(e_in, s, epochs=n , batch_size=1,verbose=1)

    prede = ctrlxplant.predict(e_in)
    p = prede.reshape((prede.shape[1], prede.shape[0]))
    outputn[t+1] = float(p[t,0])

outputn = outputn*std_dev_pos + mean_pos
setpoint = setpoint*std_dev_pos + mean_pos

time = np.arange(time_steps)*0.02

#save the trained model
ctrlxplant.save('/content/new_drive/MyDrive/ctrlxplantfinal.keras')
# load the traindmodel ( controller and plant together ) 
## please adjust the path as you need
trainedfinal = tf.keras.models.load_model('/content/new_drive/MyDrive/ctrlxplantfinal.keras') 

# Save combined model and separate controller model
ctrl_new = Sequential()
ctrl_new.add(LSTM(6, input_shape=(X1_train.shape[1], X1_train.shape[2]), return_sequences=True, name='l1'))
ctrl_new.add(Dense(1, activation='linear', name='l2'))
ctrl_new.set_weights(ctrlxplant.layers[0].get_weights())
ctrl_new.save('/content/new_drive/MyDrive/ctrl_final.keras')

# Evaluate final controller performance
time_steps = 500
setpoint1 = (0.001 - mean_pos)/std_dev_pos
E1 = np.zeros((1, time_steps, 1))
e1_in = np.zeros((1, time_steps, 1))
e1 = np.zeros((time_steps, 1))
outputn1 = np.zeros((time_steps, 1))
outputn2 = np.zeros((time_steps, 1))
s1 = np.ones_like(outputn1)*setpoint1
s1 = s1.reshape((1,s1.shape[0],s1.shape[1]))

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


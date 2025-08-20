import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

X = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
Y = np.array([[0],[1],[1],[0]],dtype=np.float32)

model = Sequential([Dense(4, activation = 'relu', input_shape = (2,)),
                    Dense(1, activation = 'sigmoid')])

model.compile(optimizer = Adam(0.1), loss = 'binary_crossentropy', metrics = ['accuracy'])

print(f"\nTraining Process Started: ")
history = model.fit(X,Y,epochs = 1000, verbose = 0)

print(f"\nAccuracy and loss at every 100th epochs: ")
for i in range(0, 1000, 100):
    print(f"Epoch {i}: loss = {history.history['loss'][i]:.4f}, accuracy = {history.history['accuracy'][i]:.4f}")
    
p = model.predict(X)
print(f"\n Final Predictions: ")
for i in range(len(X)):
    print(f"Input:{X[i]}, Pred:{int(p[i]>0.5)} (prob:{p[i][0]:.4f}).Actual:{Y[i]}")
    
print("\n Model Summary:"); model.summary()
[print(f"\n Layer {i + 1} weights:\n{w}\nBiases:\n{b}") for i, (w, b) in enumerate([l.get_weights() for l in model.layers])]

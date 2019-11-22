### Logs 

Train on 60000 samples, validate on 10000 samples

Epoch 1/20
60000/60000 [==============================] - 8s 132us/step - loss: 0.1724 - acc: 0.9485 - val_loss: 0.0958 - val_acc: 0.9715

Epoch 2/20
60000/60000 [==============================] - 6s 97us/step - loss: 0.0605 - acc: 0.9810 - val_loss: 0.1301 - val_acc: 0.9633

Epoch 3/20
60000/60000 [==============================] - 6s 97us/step - loss: 0.0496 - acc: 0.9847 - val_loss: 0.1622 - val_acc: 0.9488

Epoch 00003: ReduceLROnPlateau reducing learning rate to 0.004999999888241291.

Epoch 4/20
60000/60000 [==============================] - 6s 97us/step - loss: 0.0333 - acc: 0.9897 - val_loss: 0.0326 - val_acc: 0.9901

Epoch 5/20
60000/60000 [==============================] - 6s 97us/step - loss: 0.0319 - acc: 0.9898 - val_loss: 0.0225 - val_acc: 0.9937

Epoch 6/20
60000/60000 [==============================] - 6s 96us/step - loss: 0.0297 - acc: 0.9904 - val_loss: 0.0243 - val_acc: 0.9925

Epoch 7/20
60000/60000 [==============================] - 6s 96us/step - loss: 0.0305 - acc: 0.9903 - val_loss: 0.0318 - val_acc: 0.9898

Epoch 00007: ReduceLROnPlateau reducing learning rate to 0.0024999999441206455.

Epoch 8/20
60000/60000 [==============================] - 6s 95us/step - loss: 0.0233 - acc: 0.9925 - val_loss: 0.0205 - val_acc: 0.9932

Epoch 9/20
60000/60000 [==============================] - 6s 96us/step - loss: 0.0214 - acc: 0.9932 - val_loss: 0.0188 - val_acc: 0.9943

Epoch 10/20
60000/60000 [==============================] - 6s 97us/step - loss: 0.0199 - acc: 0.9938 - val_loss: 0.0179 - val_acc: 0.9947

Epoch 11/20
60000/60000 [==============================] - 6s 96us/step - loss: 0.0204 - acc: 0.9937 - val_loss: 0.0211 - val_acc: 0.9932

Epoch 12/20
60000/60000 [==============================] - 6s 97us/step - loss: 0.0192 - acc: 0.9940 - val_loss: 0.0203 - val_acc: 0.9940

Epoch 00012: ReduceLROnPlateau reducing learning rate to 0.0012499999720603228.

Epoch 13/20
60000/60000 [==============================] - 6s 96us/step - loss: 0.0167 - acc: 0.9947 - val_loss: 0.0155 - val_acc: 0.9957

Epoch 14/20
60000/60000 [==============================] - 6s 97us/step - loss: 0.0161 - acc: 0.9949 - val_loss: 0.0171 - val_acc: 0.9951

Epoch 15/20
60000/60000 [==============================] - 6s 97us/step - loss: 0.0148 - acc: 0.9953 - val_loss: 0.0156 - val_acc: 0.9952

Epoch 00015: ReduceLROnPlateau reducing learning rate to 0.0006249999860301614.

Epoch 16/20
60000/60000 [==============================] - 6s 96us/step - loss: 0.0139 - acc: 0.9955 - val_loss: 0.0149 - val_acc: 0.9952

Epoch 17/20
60000/60000 [==============================] - 6s 96us/step - loss: 0.0129 - acc: 0.9960 - val_loss: 0.0153 - val_acc: 0.9960

Epoch 18/20
60000/60000 [==============================] - 6s 96us/step - loss: 0.0130 - acc: 0.9961 - val_loss: 0.0148 - val_acc: 0.9957

Epoch 00018: ReduceLROnPlateau reducing learning rate to 0.0003124999930150807.

Epoch 19/20
60000/60000 [==============================] - 6s 96us/step - loss: 0.0119 - acc: 0.9961 - val_loss: 0.0145 - val_acc: 0.9957

Epoch 20/20
60000/60000 [==============================] - 6s 97us/step - loss: 0.0124 - acc: 0.9961 - val_loss: 0.0147 - val_acc: 0.9957

<keras.callbacks.History at 0x7fab233799e8>


### model.evaluate

10000/10000 [==============================] - 1s 79us/step
[0.014746979057077989, 0.9957]


### Strategy Taken

- Added a few simple functions for convolution block and transition block.
- Initial training gave good results but the train_acc, never crossed 99.5%, so experimented with reducing dropout and other regulariztion techniques.
- Added BatchNorm so that the weights could scale and shift.
- Used ReduceLROnPlateau instead of LearningRateScheduler, as it gave a better drop as training progressed (did not drop after every epoch, which helped the model train faster).
- Used a higher learning rate.
- Used lesser number of kernels initially and more later so as to express complex features better.
- Finally replaced the final 1x1 layer with GAP, and the results were even better.

### Final Validation Accuracy for Base Model

> 83.07

### Model Definition

```python
def depthwise_separable_convolution_block(no_of_kernels: int, dropout_val: float = 0.05, depth_multiplier: int = 2):
    model.add(SeparableConv2D(no_of_kernels, 3, border_mode='same', depth_multiplier=depth_multiplier))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_val))


def transition_block(no_of_output_kernels: int):
    model.add(MaxPooling2D())
    model.add(Convolution2D(no_of_output_kernels, 1))
    model.add(Activation('relu'))


model = Sequential()

model.add(SeparableConvolution2D(10, 3, input_shape=(32, 32, 3), border_mode='same', depth_multiplier=2)) # 32x32x10, 3x3
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.05))

depthwise_separable_convolution_block(16) # 32x32x16, 5x5
depthwise_separable_convolution_block(32) # 32x32x32, 7x7
depthwise_separable_convolution_block(64) # 32x32x64, 9x9
depthwise_separable_convolution_block(128) # 32x32x128, 11x11

transition_block(16) # 16x16x16, 12x12

depthwise_separable_convolution_block(16) # 16x16x16, 16x16
depthwise_separable_convolution_block(32) # 16x16x32, 20x20
depthwise_separable_convolution_block(64) # 16x16x64, 24x24
depthwise_separable_convolution_block(128) # 16x16x128, 28x28

transition_block(16) # 8x8x16, 29x29

depthwise_separable_convolution_block(16) # 8x8x16, 37x37
depthwise_separable_convolution_block(32) # 8x8x32, 45x45
depthwise_separable_convolution_block(64) # 8x8x64, 53x53
depthwise_separable_convolution_block(128) # 8x8x128, 61x61


model.add(Convolution2D(10, 1)) # 8x8x10, 61x61
model.add(GlobalAveragePooling2D()) # 10
model.add(Activation('softmax'))
```

### Epoch Logs

Epoch 1/50
390/390 [==============================] - 105s 268ms/step - loss: 1.5168 - acc: 0.4401 - val_loss: 2.5449 - val_acc: 0.3727

Epoch 2/50
390/390 [==============================] - 98s 250ms/step - loss: 1.1645 - acc: 0.5791 - val_loss: 1.4210 - val_acc: 0.5349

Epoch 3/50
390/390 [==============================] - 97s 249ms/step - loss: 1.0268 - acc: 0.6313 - val_loss: 1.2676 - val_acc: 0.5804

Epoch 4/50
390/390 [==============================] - 97s 249ms/step - loss: 0.9317 - acc: 0.6685 - val_loss: 1.1664 - val_acc: 0.6348

Epoch 5/50
390/390 [==============================] - 97s 248ms/step - loss: 0.8554 - acc: 0.6956 - val_loss: 1.2407 - val_acc: 0.6162

Epoch 6/50
390/390 [==============================] - 98s 250ms/step - loss: 0.8032 - acc: 0.7196 - val_loss: 0.7976 - val_acc: 0.7245

Epoch 7/50
390/390 [==============================] - 98s 250ms/step - loss: 0.7554 - acc: 0.7338 - val_loss: 0.8977 - val_acc: 0.7068

Epoch 8/50
390/390 [==============================] - 96s 247ms/step - loss: 0.7213 - acc: 0.7471 - val_loss: 0.9105 - val_acc: 0.7005

Epoch 9/50
390/390 [==============================] - 96s 246ms/step - loss: 0.6913 - acc: 0.7574 - val_loss: 0.7750 - val_acc: 0.7444

Epoch 10/50
390/390 [==============================] - 96s 246ms/step - loss: 0.6663 - acc: 0.7663 - val_loss: 0.8690 - val_acc: 0.7140

Epoch 11/50
390/390 [==============================] - 96s 247ms/step - loss: 0.6439 - acc: 0.7747 - val_loss: 0.8216 - val_acc: 0.7351

Epoch 12/50
390/390 [==============================] - 96s 247ms/step - loss: 0.6267 - acc: 0.7824 - val_loss: 0.8588 - val_acc: 0.7262

Epoch 00012: ReduceLROnPlateau reducing learning rate to 0.0012000000569969416.

Epoch 13/50
390/390 [==============================] - 96s 247ms/step - loss: 0.5693 - acc: 0.8014 - val_loss: 0.7421 - val_acc: 0.7547

Epoch 14/50
390/390 [==============================] - 96s 247ms/step - loss: 0.5525 - acc: 0.8090 - val_loss: 0.7032 - val_acc: 0.7655

Epoch 15/50
390/390 [==============================] - 96s 247ms/step - loss: 0.5442 - acc: 0.8112 - val_loss: 0.6178 - val_acc: 0.7917

Epoch 16/50
390/390 [==============================] - 96s 247ms/step - loss: 0.5291 - acc: 0.8157 - val_loss: 0.5456 - val_acc: 0.8129

Epoch 17/50
390/390 [==============================] - 96s 247ms/step - loss: 0.5199 - acc: 0.8194 - val_loss: 0.5410 - val_acc: 0.8158

Epoch 18/50
390/390 [==============================] - 96s 247ms/step - loss: 0.5141 - acc: 0.8202 - val_loss: 0.6151 - val_acc: 0.7946

Epoch 19/50
390/390 [==============================] - 97s 248ms/step - loss: 0.5028 - acc: 0.8235 - val_loss: 0.6124 - val_acc: 0.7972

Epoch 20/50
390/390 [==============================] - 97s 249ms/step - loss: 0.4996 - acc: 0.8237 - val_loss: 0.6766 - val_acc: 0.7790

Epoch 00020: ReduceLROnPlateau reducing learning rate to 0.000720000034198165.

Epoch 21/50
390/390 [==============================] - 97s 249ms/step - loss: 0.4656 - acc: 0.8371 - val_loss: 0.5015 - val_acc: 0.8280

Epoch 22/50
390/390 [==============================] - 97s 249ms/step - loss: 0.4567 - acc: 0.8424 - val_loss: 0.5846 - val_acc: 0.8074

Epoch 23/50
390/390 [==============================] - 97s 249ms/step - loss: 0.4505 - acc: 0.8444 - val_loss: 0.6763 - val_acc: 0.7813

Epoch 24/50
390/390 [==============================] - 97s 250ms/step - loss: 0.4504 - acc: 0.8440 - val_loss: 0.5354 - val_acc: 0.8210

Epoch 00024: ReduceLROnPlateau reducing learning rate to 0.0004320000065490603.

Epoch 25/50
390/390 [==============================] - 97s 249ms/step - loss: 0.4301 - acc: 0.8512 - val_loss: 0.4739 - val_acc: 0.8408

Epoch 26/50
390/390 [==============================] - 96s 247ms/step - loss: 0.4181 - acc: 0.8553 - val_loss: 0.5031 - val_acc: 0.8306

Epoch 27/50
390/390 [==============================] - 96s 247ms/step - loss: 0.4182 - acc: 0.8535 - val_loss: 0.5221 - val_acc: 0.8269

Epoch 28/50
390/390 [==============================] - 96s 247ms/step - loss: 0.4153 - acc: 0.8549 - val_loss: 0.4724 - val_acc: 0.8398

Epoch 29/50
390/390 [==============================] - 96s 247ms/step - loss: 0.4099 - acc: 0.8578 - val_loss: 0.4648 - val_acc: 0.8454

Epoch 30/50
390/390 [==============================] - 96s 247ms/step - loss: 0.4068 - acc: 0.8593 - val_loss: 0.4967 - val_acc: 0.8338

Epoch 31/50
390/390 [==============================] - 96s 247ms/step - loss: 0.4075 - acc: 0.8569 - val_loss: 0.4788 - val_acc: 0.8395

Epoch 32/50
390/390 [==============================] - 96s 247ms/step - loss: 0.3986 - acc: 0.8607 - val_loss: 0.4682 - val_acc: 0.8420

Epoch 00032: ReduceLROnPlateau reducing learning rate to 0.00025920000043697653.

Epoch 33/50
390/390 [==============================] - 96s 247ms/step - loss: 0.3911 - acc: 0.8649 - val_loss: 0.4866 - val_acc: 0.8362

Epoch 34/50
390/390 [==============================] - 96s 247ms/step - loss: 0.3912 - acc: 0.8623 - val_loss: 0.4717 - val_acc: 0.8391

Epoch 35/50
390/390 [==============================] - 97s 248ms/step - loss: 0.3854 - acc: 0.8654 - val_loss: 0.4614 - val_acc: 0.8460

Epoch 36/50
390/390 [==============================] - 97s 249ms/step - loss: 0.3816 - acc: 0.8661 - val_loss: 0.4896 - val_acc: 0.8362

Epoch 37/50
390/390 [==============================] - 97s 249ms/step - loss: 0.3804 - acc: 0.8681 - val_loss: 0.4315 - val_acc: 0.8576

Epoch 38/50
390/390 [==============================] - 97s 249ms/step - loss: 0.3814 - acc: 0.8659 - val_loss: 0.4571 - val_acc: 0.8463

Epoch 39/50
390/390 [==============================] - 97s 249ms/step - loss: 0.3806 - acc: 0.8679 - val_loss: 0.4556 - val_acc: 0.8503

Epoch 40/50
390/390 [==============================] - 98s 250ms/step - loss: 0.3760 - acc: 0.8689 - val_loss: 0.4647 - val_acc: 0.8446

Epoch 00040: ReduceLROnPlateau reducing learning rate to 0.00015551999676972626.

Epoch 41/50
390/390 [==============================] - 97s 248ms/step - loss: 0.3703 - acc: 0.8704 - val_loss: 0.4559 - val_acc: 0.8467

Epoch 42/50
390/390 [==============================] - 96s 246ms/step - loss: 0.3685 - acc: 0.8708 - val_loss: 0.4765 - val_acc: 0.8410

Epoch 43/50
390/390 [==============================] - 96s 246ms/step - loss: 0.3631 - acc: 0.8728 - val_loss: 0.4755 - val_acc: 0.8402

Epoch 00043: ReduceLROnPlateau reducing learning rate to 9.331199980806559e-05.

Epoch 44/50
390/390 [==============================] - 96s 247ms/step - loss: 0.3626 - acc: 0.8719 - val_loss: 0.4414 - val_acc: 0.8543

Epoch 45/50
390/390 [==============================] - 96s 247ms/step - loss: 0.3572 - acc: 0.8748 - val_loss: 0.4547 - val_acc: 0.8504

Epoch 46/50
390/390 [==============================] - 96s 246ms/step - loss: 0.3566 - acc: 0.8746 - val_loss: 0.4461 - val_acc: 0.8529

Epoch 00046: ReduceLROnPlateau reducing learning rate to 5.598720163106918e-05.

Epoch 47/50
390/390 [==============================] - 96s 247ms/step - loss: 0.3566 - acc: 0.8745 - val_loss: 0.4531 - val_acc: 0.8496

Epoch 48/50
390/390 [==============================] - 96s 247ms/step - loss: 0.3563 - acc: 0.8753 - val_loss: 0.4439 - val_acc: 0.8535

Epoch 49/50
390/390 [==============================] - 96s 247ms/step - loss: 0.3538 - acc: 0.8760 - val_loss: 0.4512 - val_acc: 0.8503

Epoch 00049: ReduceLROnPlateau reducing learning rate to 3.359232141519897e-05.

Epoch 50/50
390/390 [==============================] - 97s 249ms/step - loss: 0.3524 - acc: 0.8765 - val_loss: 0.4514 - val_acc: 0.8497
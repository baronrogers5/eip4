### The test data accuracy is 0.991

1. **Convolution** - The process through which a kernel extracts information from a channel. It involves multiplying the kernel values with the channel and adding up the products. The same step is repeated as the kernel slides over the complete channel.

2. **Filters/Kernels** - These are feature extractors. They are trainable parameters that the neural network learns during training. They are stacked on top of one another until the required receptive field is attained. In general we prefer the 3x3 kernel.

3. **Epochs** - A pass through the complete dataset is called an epoch. After an epoch the network has seen all the images. In general an epoch is broken down into iterations where batches of data are processed. This is done due to limitation in GPU memory capacity.

4. **1x1 Convolution** - A 1x1 convolution is a feature combiner, it is used to reduce the number of channels, and it does not alter the shape of the image in any way. It is a useful operator as it combines relevant information after the features of an image have been extracted.

5. **3x3 Convolution** - The most important kernel, it is used to extract information/features from images/channels. A 3x3 can be stacked with another 3x3 to get a receptive field of 5x5, with fewer trainable parameters than using a 5x5 kernel. 

6. **Feature Maps** - A feature map or channel is a collection of similar features of an image. Feature maps are generated after a kernel convolves over a channel. The number of channels of a kernel that convolves with an image must equal the number of feature maps of that image.

7. **Activation Function** - A neural network is a complex function that maps the input to the output. The translation from input to output is generally non-linear in nature. But convolution, the main operator used by the networks is a linear operation. Activation functions provide the necessary non-linearity to our networks to learn the deep intricacies. 

8. **Receptive Field** - The grid of pixels the network has seen upto a layer, defines the receptive field of that layer. The receptive field grows as the size of a network increases. In general, we only build networks whose receptive field is the size of the objects we want to detect.

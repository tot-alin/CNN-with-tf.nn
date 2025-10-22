# Convolutional neural network 2D with module tf.nn

  This project aims at realizing a model based on 2D convolutional neural networks
using the tf.nn module in TensorFlow. The model has the ability to classify eight items
['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person'] and has been trained with
6899 images. The description of this project is mainly focused on describing the
functionality of 2D convolutional neural networks. 
<p align="center">
<img width="833" height="374" alt="Screenshot from 2025-10-22 16-25-31" src="https://github.com/user-attachments/assets/e6ebe532-65b1-4b4e-8a1c-87129d3c2f4c" />
</p>

## Elementary functions in the structure of CNN models
In most cases, CNN models contain at least three important elements, described below.
* CNN
* Pool
* Activate function


## Convolutional neural networks
Convolutional Neural Networks (CNNs) are an advanced version of Artificial Neural Networks (ANNs), mainly designed to extract features from matrix datasets. These types of networks are used primarily to deal with video, audio, and text content databases. As illustrated in Figure 1.1, the components of a simple 2-dimensional CNN are: input data matrix, filter, and map. 

<br />The feature detector, i.e., the filter, is a two-dimensional matrix of weights that calculates over the entire area of the matrix containing the input data. The pitch of the filter must be defined on both horizontal and vertical axes. If the filter step is greater than 1, the size of the output matrix Map will be smaller than the input matrix. The elements of the Map matrix are the result of the weighted sum of the filter matrix and the area of the input matrix where the filter maps. So, each step of the filter results in one element in the Map matrix. An important component to note is that the value of the weights in the filter matrix remains constant throughout the time that the filter is applied to the area of the input matrix.

<br />In the case of a 3-channel (RGB) image, we have a 3-dimensional matrix as input data, and the approach is similar to the previous description. Figure 1.2 illustrates the workflow of a CNN with a 3-dimensional (RGB) matrix as input and a 3-dimensional matrix as output (i.e., n Map matrices). In this case, n filters are required, corresponding to n outputs. Each filter consists of 3 2D matrices that are culled on the corresponding matrix in the RGB matrix, resulting in 3 matrices (YR, YG, YB) where each element is the weighted sum of the filter displacements. Each 'n' Map matrix is the result of the sum of the corresponding (n) matrices YR, YG, and YB. Equation 1 presents in analytic form the above description, and one can observe the functionality much more precisely.

Fig. 1.1 One core CNN with 2D data input and one map output
<p align="center">
<img width="520" height="185" alt="image1" src="https://github.com/user-attachments/assets/20d67388-2416-4ae4-b00c-ed24eaa3108a" />
</p>
Fig. 1.2 One core CNN with 3D data input and n maps output
<p align="center">
<img width="602" height="514" alt="Screenshot from 2025-10-22 16-42-37" src="https://github.com/user-attachments/assets/66d3806f-721e-4ff0-aeae-000519a0f081" />
</p>
<br />With the tf.nn.conv2d() method, we can compute a 2D convolution using a matrix with 4D input = [batch, height, width, channels] as input and a filter (tensor). In general, the filter is realized as a matrix with random elements, with the form filter = [filter_height, filter_width, no. in_channels, no. out_channels]. If data_format = 'NHWC' the filter displacement will be strides = [batch, in_height, in_width, in_channels]. The mode of the displacement on the edge of the matrix is assigned as padding= 'VALID' or 'SAME', i.e. the filter edge not to exceed the matrix or the center of the filter to reach the edge of the metric on which it is culled, adding 0's on the edge of the input matrix. Method expression: tf.nn.conv2d(input, filter, strides, padding, data_format=None, name=None)
<br />.
<p align="center">
<img width="1068" height="324" alt="image" src="https://github.com/user-attachments/assets/5f942a25-3855-46b1-bff6-0aba9adb4577" />
</p>

<p align="center">
<img width="720" height="124" alt="image" src="https://github.com/user-attachments/assets/e931dccd-da2b-426e-b3ce-1e7f02504d4d" />
</p>
<p align="center">
<img width="750" height="124" alt="image" src="https://github.com/user-attachments/assets/de17a1c6-fe8f-4ded-86bd-353ffa1b6eac" />
</p>
<p align="center">
<img width="733" height="124" alt="image" src="https://github.com/user-attachments/assets/20a11b68-c123-47b6-94b8-ca83a90642c3" />
</p>

The result has the following form:
<p align="center">
<img width="429" height="194" alt="image" src="https://github.com/user-attachments/assets/ddcb002c-2421-4109-93fc-bd659c993f3c" />
</p>



## Activate function


## Influența CNN, Pool și Activate function, asupra imagini



## Bibliography :

https://www.datacamp.com/tutorial/introduction-to-convolutional-neural-networks-cnns
https://www.simplilearn.com/tutorials/deep-learning-tutorial/convolutional-neural-network
https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/
https://www.datacamp.com/blog/rectified-linear-unit-relu
https://www.geeksforgeeks.org/deep-learning/cnn-introduction-to-pooling-layer/
https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/

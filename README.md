# Convolutional neural network 2D with module tf.nn

  This project aims at realizing a model based on 2D convolutional neural networks
using the tf.nn module in TensorFlow. The model has the ability to classify eight items
['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person'] and has been trained with
6899 images. The description of this project is mainly focused on describing the
functionality of 2D convolutional neural networks. 
<p align="center">
<img width="833" height="374" alt="Screenshot from 2025-10-22 16-25-31" src="https://github.com/user-attachments/assets/e6ebe532-65b1-4b4e-8a1c-87129d3c2f4c" />
</p>
<br />
<br />

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
<br />.
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
<br />
<br />

## Pool
The pooling layer is used in the structure of CNNs in order to reduce the size of feature maps while preserving the relevant information. The functionality of this method (Fig. 2.1) is based, as before, on a two-dimensional filter that slides on the feature map channel.
<br /> Fig. 2.1 Pool
<p align="center">
<img width="595" height="176" alt="Screenshot from 2025-10-22 17-26-27" src="https://github.com/user-attachments/assets/2eeaefa4-26d9-405a-bc1a-7695df560e55" />
</p>
For example, the max_plot method takes the maximum value from the area it maps to. Setting the functionality of the tf.nn.max_pool() method involves specifying the window shape, ksize=[corresponding pool number, filter height, filter width, channel number], and the filter's shift steps. Padding the filter to the edge of the feature matrix padding= 'VALID' or 'SAME', (VALID) meaning that the filter edge does not exceed the matrix it slides on or (SAME) the center of the filter reaches the edge of the metric it slides on, adding 0 - ions to the edge of the matrix.
<br />
<br />

## Activate function
The activation function is meant to transform the linear form of the weighted sum to a nonlinear form, i.e., the output does not change proportionally to the input. In most real-world situations, the data have complex shapes, and by adding a nonlinear function such as ReLU, Sigmoid, or Tanh, the network can create curved decision boundaries to separate them correctly. A commonly used function is the function ReLU, with the relation f(x)=max(0,x).
<br />
<br />

## The influence of CNN, Pool and Activate function on images during processing
In the following, a series of images, taken from the attached program, are presented, and the recursion of image modifications can be observed.
<p align="center">
<br />original image (224X224)
<br /><img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/12b110f4-3389-47af-9185-82a99679adf0" />
</p>

<p align="center">
<br />convolutions 1  (224, 224, 8)
<br /><img width="780" height="383" alt="image" src="https://github.com/user-attachments/assets/b0a26ad9-5413-4baf-85cb-c1c3e7703499" />
</p>

<p align="center">
<br />activate 1  (224, 224, 8)
<br /><img width="781" height="377" alt="image" src="https://github.com/user-attachments/assets/e40efe09-d708-4904-87ea-2caa097bebdf" />
</p>

<p align="center">
<br />pool 1  (112, 112, 8)
<br /><img width="780" height="376" alt="image" src="https://github.com/user-attachments/assets/400e69f6-1e86-4fba-95ad-db3ac9f4c07e" />
</p>

<p align="center">
<br />convolutions 2  (112, 112, 16)
<br /><img width="774" height="775" alt="image" src="https://github.com/user-attachments/assets/effc898e-f32d-4674-8920-bb14df71725f" />
</p>

<p align="center">
<br />activate 2  (112, 112, 16)
<br /><img width="778" height="775" alt="image" src="https://github.com/user-attachments/assets/58383146-7dd9-4b8e-85b9-44575a1406c2" />
</p>

<p align="center">
<br />pool 2  (56, 56, 16)
<br /><img width="777" height="781" alt="image" src="https://github.com/user-attachments/assets/6b15c568-4368-4cd5-b0a2-6ad1f2abfb39" />
</p>

<p align="center">
<br />convolutions 3  (56, 56, 32)
<br /><img width="699" height="775" alt="image" src="https://github.com/user-attachments/assets/e035577e-9015-439e-a571-70841f3b71de" />
</p>

<p align="center">
<br />activate 3  (56, 56, 32)
<br /><img width="694" height="777" alt="image" src="https://github.com/user-attachments/assets/b2ed6081-dbfd-4c63-911a-56312bf4bac9" />
</p>

<p align="center">
<br />pool 3  (28, 28, 32)
<br /><img width="698" height="775" alt="image" src="https://github.com/user-attachments/assets/51979a33-9e16-4c3b-968d-7fce54a5cc3f" />
</p>
<br />
<br />


## Bibliography :

https://www.datacamp.com/tutorial/introduction-to-convolutional-neural-networks-cnns
https://www.simplilearn.com/tutorials/deep-learning-tutorial/convolutional-neural-network
https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/
https://www.datacamp.com/blog/rectified-linear-unit-relu
https://www.geeksforgeeks.org/deep-learning/cnn-introduction-to-pooling-layer/
https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/


# Deep Convolutional Neural Network on CIFAR-10

## Overview
This repository demonstrates the design, implementation, and evaluation of a deep convolutional neural network (CNN) for image classification on the CIFAR-10 dataset using Tensorflow 2.5 and its high-level Keras API. The CIFAR-10 dataset is a staple in computer vision, consisting of 60,000 32x32 color images distributed across 10 distinct classes.

![CIFAR-10 Image](https://paperswithcode.com/media/datasets/CIFAR-10-0000000431-b71f61c0_U5n3Glr.jpg)

## Techniques and Methods
- **Deep Convolutional Neural Networks (CNNs)**: CNNs are a class of deep neural networks particularly effective for tasks related to image perception. They can learn hierarchical features from images, where initial layers learn basic features (like edges) and deeper layers learn more complex patterns.
  
- **Data Augmentation**: To avoid overfitting and increase the diversity of the training dataset, data augmentation techniques such as rotation, zoom, and horizontal flipping are employed.
  
- **Batch Normalization**: Used to normalize the activations of a given input volume before passing it to the next layer. It helps in faster convergence and overall stability of the network.
  
- **Softmax Activation**: In the final layer, a softmax activation function is used to output a probability distribution over the 10 classes.

## Notebook Structure
1. **Introduction**: Brief introduction to the CIFAR-10 dataset.
2. **Dataset Overview**: Insights into the dataset, its classes, and structure.
3. **Data Preprocessing**: Normalization of pixel values and implementation of data augmentation techniques.
4. **Model Design and Implementation**: Crafting the CNN architecture tailored for the CIFAR-10 dataset. Details on the layers used, activation functions, and architectural choices.
5. **Model Compilation**: Setup with appropriate loss function (categorical crossentropy), optimizer (like Adam), and performance metrics.
6. **Model Training**: Training process including batch size, epochs, and validation split details.
7. **Model Evaluation**: Metrics used for evaluation, such as accuracy, precision, recall, and the confusion matrix.

## Usage
1. Clone this repository.
2. Install the required libraries, especially Tensorflow 2.5+.
3. Execute the `Modified_CNN_CIFAR10.ipynb` notebook.

## Contributing
Contributions to improve the model or any other aspect are welcome! Please fork this repository and submit a pull request.

## License
MIT License.


# VIVA Notes

The main difference between Autoencoders and Principle Component Analysis (PCA) is that while PCA finds the directions along which you can project the data with maximum variance, Autoencoders reconstruct our original input given just a compressed version of it.


Sure, here's an updated table that includes what each type of autoencoder is typically used for:

| Type of Autoencoder | Description | Typical Use Cases |
| --- | --- | --- |
| Linear Autoencoder | A type of autoencoder that uses a linear activation function in the output layer. | Feature extraction, dimensionality reduction, data compression |
| Undercomplete Autoencoder | A type of autoencoder that has a bottleneck layer that is smaller than the input layer, forcing the network to learn a compressed representation of the data. | Feature extraction, dimensionality reduction, data compression |
| Denoising Autoencoder | A type of autoencoder that is trained to reconstruct the original input data from noisy or corrupted data. | Image and signal denoising, image and signal inpainting, anomaly detection |
| Sparse Autoencoder | A type of autoencoder that learns a compressed representation of the data by enforcing sparsity in the activations of the hidden layer. | Feature extraction, anomaly detection, image and signal classification |
| Contractive Autoencoder | A type of autoencoder that is trained to learn a compressed representation of the data by adding a penalty term to the loss function that measures the sensitivity of the hidden layer to small changes in the input. | Feature extraction, data compression, image and signal classification |


Sure! Here's a table that provides information about different deep learning models/algorithms, including their type, architecture, components, and applications:

| Model/Algorithm | Type | Architecture | Components | Application |
|-----------------|------|--------------|------------|-------------|
| Perceptron | Supervised | Single-layer neural network | Consists of input nodes, weights, activation function, and output node | Binary classification, linearly separable problems |
| Multilayer Perceptron (MLP) | Supervised | Feedforward neural network with multiple hidden layers | Consists of input nodes, hidden layers with weights and activation functions, output layer | Classification, regression, complex non-linear problems |
| Stochastic Gradient Descent (SGD) | Supervised | Optimization algorithm for training neural networks | Iterative optimization method that updates weights based on a subset of training data | Training deep neural networks, optimization |
| Backpropagation | Supervised | Learning algorithm for training neural networks | Computes gradients of the loss function with respect to the weights, used in SGD for weight updates | Training deep neural networks, optimization |
| Deep Neural Network (DNN) for Classification | Supervised | Multiple layers of interconnected nodes, typically with ReLU or sigmoid activations | Consists of input layer, multiple hidden layers with weights and activation functions, output layer with softmax activation | Classification tasks, image recognition, speech recognition |
| Autoencoder | Unsupervised | Neural network with encoder and decoder components | Encoder compresses input data to lower-dimensional representation, decoder reconstructs original data | Dimensionality reduction, feature extraction, denoising, anomaly detection |
| Convolutional Neural Network (CNN) | Supervised | Specialized neural network for processing grid-like data (e.g., images) | Consists of convolutional layers for feature extraction, pooling layers for downsampling, fully connected layers for classification | Image recognition, object detection, video analysis |
| Long Short-Term Memory (LSTM) | Supervised | Recurrent neural network (RNN) architecture | Contains memory cells with gating mechanisms for capturing long-term dependencies | Sequence data processing, natural language processing, speech recognition |

Note: Deep learning models and algorithms can be used for a variety of applications beyond those listed, and the architecture and components may vary depending on specific implementations and variations of these models/algorithms.


Sure! Here's a table that provides information about different deep learning models/algorithms, including their type, architecture, components, and applications:

| Model/Algorithm | Type | Architecture | Components | Application |
|-----------------|------|--------------|------------|-------------|
| Perceptron for AND gate | Supervised | Single-layer neural network with a single output neuron | Input layer with two input nodes, output layer with one output node | Simulating the AND gate logic with binary inputs and outputs |
| Multilayer Perceptron for XOR gate | Supervised | Multilayer neural network with one or more hidden layers | Input layer with two input nodes, one or more hidden layers with variable number of nodes, output layer with one output node | Simulating the XOR gate logic with binary inputs and outputs |
| Stochastic Gradient Descent (SGD) | Unsupervised | Optimization algorithm for training deep neural networks | Iterative optimization algorithm that updates model parameters based on a subset of training data at each iteration | Optimization of deep neural networks for various tasks |
| Backpropagation | Supervised | Learning algorithm for training deep neural networks | Computes gradients of the loss function with respect to the model parameters and updates them accordingly | Training deep neural networks for various tasks |
| Deep Neural Network (DNN) for classification | Supervised | Multilayer neural network with one or more hidden layers, typically with many nodes | Input layer with features as input nodes, one or more hidden layers with many nodes, output layer with number of nodes equal to the number of classes | Classification tasks, such as image classification, speech recognition, etc. |
| Autoencoder for image compression | Unsupervised | Neural network with an encoder and decoder architecture | Encoder compresses input images into a lower-dimensional representation, decoder reconstructs the original images from the compressed representation | Image compression for reducing storage and transmission requirements |
| Autoencoder for image denoising | Unsupervised | Neural network with an encoder and decoder architecture | Encoder maps noisy input images to a lower-dimensional representation, decoder reconstructs denoised images from the compressed representation | Image denoising for removing noise from images |
| Convolutional Neural Network (CNN) for handwritten digit recognition | Supervised | Neural network with convolutional and pooling layers followed by one or more fully connected layers | Convolutional layers for feature extraction, pooling layers for spatial downsampling, fully connected layers for classification | Handwritten digit recognition for applications such as digit recognition in postal codes, bank checks, etc. |
| Convolutional Neural Network (CNN) for image classification | Supervised | Neural network with convolutional and pooling layers followed by one or more fully connected layers | Convolutional layers for feature extraction, pooling layers for spatial downsampling, fully connected layers for classification | Image classification tasks, such as object recognition, scene recognition, etc. |
| Long Short-Term Memory (LSTM) for sentiment analysis | Supervised | Recurrent neural network (RNN) with LSTM cells for sequential data processing | LSTM cells for capturing long-term dependencies in sequential data, fully connected layers for classification | Sentiment analysis for determining sentiment or emotion in text data, such as social media posts, customer reviews, etc. |

Note: The architectures and components mentioned above are high-level representations and may vary depending on the specific implementation and variations of the models/algorithms.

Sure! Here's a table that provides information about different activation functions, including their equations, pros, cons, and a short summary:

| Activation Function | Equation | Pros | Cons | Summary |
|---------------------|----------|------|------|---------|
| Linear | f(x) = x | Simple and preserves linearity. | Limited representation power, can't model complex non-linearities. | Simple and preserves linearity, but limited in modeling complex non-linearities. |
| Sigmoid | f(x) = 1 / (1 + exp(-x)) | Squashes output between 0 and 1, suitable for binary classification. | Prone to vanishing gradients, not suitable for deep networks. | Smooth and maps input to a probability-like output, but prone to vanishing gradients and not suitable for deep networks. |
| Rectified Linear Unit (ReLU) | f(x) = max(0, x) | Overcomes vanishing gradient problem, computationally efficient. | Can suffer from "dying ReLU" problem, not suitable for negative inputs. | Simple and computationally efficient, overcomes vanishing gradient problem. However, can suffer from "dying ReLU" problem and not suitable for negative inputs. |
| Leaky ReLU | f(x) = max(αx, x), where α is a small positive constant | Overcomes "dying ReLU" problem, allows for negative inputs. | Requires tuning of α hyperparameter. | Overcomes "dying ReLU" problem and allows for negative inputs, but requires tuning of α hyperparameter. |
| Hyperbolic Tangent (tanh) | f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) | Maps input to a range between -1 and 1, suitable for symmetric activation. | Prone to vanishing gradients for extreme values, not zero-centered. | Similar to sigmoid but maps input to a range between -1 and 1. Prone to vanishing gradients and not zero-centered. |
| Softmax | f(x_i) = exp(x_i) / ∑(exp(x_j)), for all i | Outputs probabilities for multi-class classification, suitable for final output layer. | Sensitive to input scale, not suitable for intermediate layers. | Converts input to probabilities for multi-class classification, but sensitive to input scale and not suitable for intermediate layers. |

Note: The choice of activation function depends on the specific problem, network architecture, and dataset, and experimentation is important to determine the best activation function for a given task.

Sure! Here's an updated table that includes a short summary for each optimizer:

| Optimization Algorithm | Equation | Pros | Cons | Summary |
|------------------------|----------|------|------|---------|
| Gradient Descent (GD) |  θ = θ - α ∇J(θ) | Simple and easy to implement. | Can be slow for large datasets or complex models. Prone to getting stuck in local optima. | Classic optimization algorithm, but can be slow for large datasets or complex models. |
| Stochastic Gradient Descent (SGD) | θ = θ - α ∇J(θ; xi, yi) | Faster convergence, good for large datasets. | High variance due to random sampling, can have slower convergence. | Faster than GD, but has high variance due to random sampling. Good for large datasets. |
| Mini-Batch Gradient Descent | θ = θ - α ∇J(θ; X_batch, y_batch) | A compromise between GD and SGD, balances speed and stability. | Requires tuning of batch size. | A balance between GD and SGD, provides speed and stability, but requires tuning of batch size. |
| Momentum-based Gradient Descent | v = βv - α ∇J(θ), θ = θ + v | Accelerates convergence, helps escape local optima. | Requires tuning of momentum hyperparameter. | Accelerates convergence and helps escape local optima by introducing momentum. Requires tuning of momentum hyperparameter. |
| Nesterov Accelerated Gradient Descent (NAG) | v = βv - α ∇J(θ + βv), θ = θ + v | Improves over regular momentum, more accurate updates. | Requires tuning of momentum and β hyperparameters. | An improvement over regular momentum that provides more accurate updates by looking ahead. Requires tuning of momentum and β hyperparameters. |
| AdaGrad | θ = θ - α / √(∑g^2), g = ∇J(θ) | Adapts learning rates for each parameter, works well for sparse data. | Can result in very small learning rates, causing slow convergence. | Adapts learning rates for each parameter based on past gradients, but can result in very small learning rates. Works well for sparse data. |
| Adam | m = β1m + (1-β1)∇J(θ), v = β2v + (1-β2)(∇J(θ))^2 , θ = θ - α m / (√v + ε) | Adaptive learning rates, handles sparse data and noise. | Requires tuning of multiple hyperparameters. | Adaptive Moment Estimation (Adam) adjusts learning rates based on past gradients and moment estimates. Handles sparse data and noise, but requires tuning of multiple hyperparameters. |
| RMSProp | v = βv + (1-β)(∇J(θ))^2 , θ = θ - α ∇J(θ) / (√v + ε) | Adapts learning rates, performs well in non-stationary settings. | Requires tuning of hyperparameters. | Adapts learning rates based on the magnitude of past gradients. Performs well in non-stationary settings, but requires tuning of hyperparameters. |

Note: The summary column provides a brief overview of each optimization algorithm, highlighting its strengths and limitations. However, the performance of each optimizer may vary depending on the specific problem and dataset, and proper hyperparameter tuning and experimentation is important to determine the best optimizer for a given task.


Sure! Here's a table that summarizes various regularization techniques and the problems they help address:

| Regularization Technique | Problem it Solves | Description |
|------------------------|-------------------|-------------|
| L1 regularization      | Overfitting       | Adds a penalty to the absolute values of the model's weights during training, encouraging sparsity and reducing the complexity of the model. |
| L2 regularization      | Overfitting       | Adds a penalty to the squared values of the model's weights during training, encouraging small weights and preventing over-reliance on individual features. Also known as weight decay. |
| Parameter sharing      | Overfitting, Underfitting | Shares parameters across different parts of the model, such as in convolutional neural networks (CNNs), to reduce the risk of overfitting and improve generalization. |
| Dropout                | Overfitting       | Randomly drops out a fraction of the neurons during training, preventing them from contributing to the model's output. This helps to prevent overfitting and encourages the model to learn more robust representations. |
| Weight Decay           | Overfitting       | Adds a penalty term to the loss function during training, which discourages large weights and helps to prevent overfitting. It is often implemented using L2 regularization. |
| Batch Normalization    | Overfitting       | Normalizes the inputs to a layer during training by scaling and shifting them, which helps to stabilize and accelerate training. It can also act as a form of regularization by reducing internal covariate shift. |
| Early Stopping         | Overfitting       | Monitors the model's performance on a validation set during training and stops training early when the performance on the validation set starts to degrade, preventing overfitting. |
| Data Augmentation      | Overfitting       | Generates new training data by applying various transformations, such as rotation, scaling, or flipping, to the original data. This helps to increase the size and diversity of the training set, preventing overfitting. |
| Adding Noise to input & output | Overfitting       | Adds random noise to the input or output of the model during training, which helps to regularize the model by making it more robust to small perturbations in the input or output data. |

Note: Regularization techniques are often used in combination to achieve better performance and prevent overfitting or underfitting in machine learning models. The choice of regularization technique(s) to use depends on the specific problem, data, and model architecture being used.

In deep learning, regularization techniques such as Dropout, Early stopping, and Data augmentation can help address underfitting problems:

Dropout: Dropout is a regularization technique where during training, randomly selected neurons are ignored, or "dropped out," with a certain probability. This helps in preventing the model from relying too heavily on specific neurons and encourages the learning of more robust features, which can help mitigate underfitting.

Early stopping: Early stopping is a regularization technique that stops the training process before it reaches the maximum number of epochs if the performance on a validation set starts to degrade. This helps in preventing the model from overfitting the training data and improves its ability to generalize, which can also help address underfitting by stopping the training process at an appropriate point.

Data augmentation: Data augmentation is a regularization technique where the training data is artificially augmented by applying various transformations, such as rotation, scaling, and flipping, to generate additional training samples. This helps in increasing the diversity and quantity of the training data, which can help the model learn more robust features and overcome underfitting by exposing it to more variations in the data.

In summary, regularization techniques such as Dropout, Early stopping, and Data augmentation can help address underfitting problems in deep learning models by encouraging the learning of more robust features, stopping the training process at an appropriate point, and increasing the diversity of training data, respectively.

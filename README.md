# ML-Algorithms

1. ## Perceptron

   This code demonstrates the implementation of the Perceptron algorithm from scratch and its application to different datasets using NumPy and matplotlib. Below is an explanation of each part of the code:
   
   **Part 1: Linearly Separable Data (make_blobs)**
   
   1. The code imports necessary libraries and generates a synthetic dataset using `make_blobs` from `sklearn.datasets`.
   2. The Perceptron algorithm is implemented from scratch using NumPy.
   3. The dataset is augmented to include a bias term.
   4. The algorithm runs for ten iterations, and intermediate hyperplanes are shown for each iteration.
   5. Finally, the final decision boundary is plotted after convergence.
   
   **Part 2: Linearly Separable Data with Batch Gradient Descent**
   
   1. This part repeats the Perceptron algorithm, but this time, it uses batch gradient descent with a specified step size (learning rate) to update the weights.
   2. The algorithm is again run for ten iterations, and the error is printed for each iteration.
   3. The final decision boundary is plotted after convergence.
   
   **Part 3: Non-Linearly Separable Data (make_circles)**
   
   1. This part generates another synthetic dataset using `make_circles` from `sklearn.datasets`.
   2. Since the data is not linearly separable, the feature space is augmented to include second-order features (terms up to degree 2).
   3. The Perceptron algorithm is applied to find the decision boundary in the augmented feature space.
   4. The final decision boundary is plotted in the original two-dimensional space using `contour()`.
   
   **Part 4: Test Accuracy**
   
   1. In this part, the code repeats the Perceptron algorithm for both linearly separable and non-linearly separable datasets, each containing 1000 points.
   2. The datasets are split into 50% training and 50% test sets.
   3. The Perceptron model is trained on the training data and tested on the test data.
   4. The accuracy of the model on the test data is calculated and reported for both datasets.

2. ## K-Means Clustering Algorithm:
   - This algorithm performs K-Means clustering on a given dataset.
   - The `KMeansCluster` class is defined, which takes the number of clusters (`K`) and maximum iterations (`iterations`) as parameters.
   - The `fit` method fits the data and finds the clusters.
   - The `initializeCentroids` method initializes the centroids randomly.
   - The `createCluster` method assigns data points to their closest centroids and forms clusters.
   - The `updateCentroids` method updates the centroids based on the cluster data points.
   - The `predictCluster` method assigns data points to the closest cluster.
   - The `check` method checks if the centroids have converged.
   - The `plotGraph` method plots the data points and centroids.
   - The `cost_function` method calculates the cost function for the clusters.
   - A sample K-Means clustering is performed on hypothetical data, and the results are plotted.


3. ## Gaussian Mixture Model (GMM) Algorithm:
   - This algorithm implements Gaussian Mixture Model (GMM) for clustering.
   - The `GMM` class is defined, which takes the number of clusters (`clusters`), maximum iterations (`max_iter`), and tolerance (`tol`) as parameters.
   - The `fit` method fits the GMM to the data.
   - The `intializeParameters` method initializes the GMM parameters (means, covariances, and priors).
   - The `estimationStep` method performs the E-Step of the EM algorithm, updating the posteriors.
   - The `maximizationStep` method performs the M-Step of the EM algorithm, updating the parameters.
   - The `gaussianDensityFunction` method calculates the probability density of a data point given a Gaussian distribution.
   - The `predict` method predicts the clusters for new data points.
   - A sample GMM clustering is performed on hypothetical data, and the results are plotted.

4. ## Naive Bayes Classifier:
   - A Naive Bayes Classifier is implemented for binary classification.
   - The `likelihoodRatio` function calculates the likelihood ratio of class probabilities given the data.
   - The `kvalue` function calculates the threshold value for classification based on error rate.
   - The `npClassifier` function classifies data points based on the likelihood ratio and threshold.
   - The `calulateMetrics` function calculates true positives, false positives, and false negatives.
   - The classifier is tested on hypothetical data, and metrics (accuracy, true positives, false positives, etc.) are calculated.

5. ## ROC Curve for Classifier:
   - The code generates data from two classes and computes the likelihood ratio function.
   - It then calculates true and false positive rates for various thresholds.
   - The ROC curve is plotted using matplotlib.

6. ## Maximum-Minimum Classifier:
   - A Maximum-Minimum Classifier is implemented for binary classification.
   - The `maxMinClassifier` function classifies data points based on each class's maximum and minimum values.
   - The classifier is tested on hypothetical data, and predictions are printed.

7. ## MinMaxClassifier Class:
   - A `MinMaxClassifier` class is defined for binary classification.
   - The `fit` method fits the classifier to the data.
   - The `predict` method predicts the class of new data points based on the fitted classifier.
   - The classifier is tested on hypothetical data, and accuracy is calculated.



# Neural-Network

This code defines a neural network model and demonstrates its training process on the Boston housing dataset using Stochastic Gradient Descent (SGD). The neural network consists of multiple layers with an activation function and weights. The primary goal is to predict house prices using the given features.

Here's a breakdown of the code:

1. `Layers` class: Represents a single layer in the neural network. It takes the forward and backward activation functions, the number of output neurons, the number of inputs, and seed for random initialization. It initializes the weights and biases randomly.

2. `NeuralNetwork` class: Represents the neural network model. It defines the neural network architecture as a list of layer objects and a loss object (mean square error or cross-entropy loss). It has methods for forward pass (calculating output) and backward pass (updating weights and biases based on the loss).

3. `mean_square_error` class: Represents the mean squared error loss function, which calculates the error between predicted and actual outputs.

4. `cross_entropy_loss` class: Represents the cross-entropy loss function, which measures the difference between predicted and actual outputs for multi-class classification problems.

5. Activation functions: The code defines several activation functions, such as linear, sigmoid, and tanh (both forward and backward passes).

6. Boston dataset: The code loads the Boston housing dataset from sklearn and preprocesses it by normalizing the features and target.

7. Training the neural network: The code sets up different configurations for the neural network by creating layers with different activation functions and connecting them in various ways. It then performs Stochastic Gradient Descent (SGD) to train the neural network on the Boston housing dataset. The training process includes forward and backwards passes to update the weights and biases for each layer.

8. Error Visualization: The code plots the error at each iteration during training, showing how the loss decreases over time.

9. Model evaluation: The code calculates the mean squared error and R-squared value to evaluate the performance of the trained model.

10. Accuracy testing: The code defines a function to test the model's accuracy on the test data.





The goal is to train a CNN to recognize images of digits from the MNIST dataset. This dataset consists of
gray level images of size 28x28. There is a standard partitioning of the dataset into training and testing.
The training has 60,000 examples and the testing has 10,000 examples. Standard CNN models achieves over
99% accuracy on this dataset.

Constraints:
In this project you are asked to solve a similar problem of creating a classifier for the MNIST data. However,
the training data in our case has only 6,000 images, and each image is shrunk to size 7x7. Specifically,
your program must include the following:

1. Your program must set the random seeds of python and tensorflow to 1 to make sure that your results
are reproducible.
2. The first layer in the network must be a 4 ⇥ 4 maxpooling layer. This e↵ectively shrinks the images
from 28x28 to 7x7.
3. Your program will be tested by training on a fraction of 0.1 of the standard training set. The testing
data will be the entire standard testing set.
4. The training and testing in you program should not take more than 6 minutes.

Implementation:
The images were preprocessed using erosion technique and a residual network was designed for this purpose. The model achieved an accuracy of over 90% in 4 and 1/2 minutes over different training datasets.

# cadoAI

This is yet another CNN for making a choice between a dog and a cat made on **PyTorch**

## The model

The model is a traditional Convolutional Neural Network with 3 convolutional layers and 1 fully connected layer. 
First layer takes a 64 by 64 RGB image and applies 32 filters with kernel size 3, stride 1 and padding 1. 
Pooling layer creates 32 feature maps and passes it to the next convolutional layer.

Second layer - convolution and poolign creates 64 feature maps from the previous 32 ones. 
Third layer - convolution and pooling creates 128 feature maps from the previous 64 ones. 

Rectifying: getting ready for passing feature maps into a fooling connected layer means rectifying all of them into one column.

Fully connected layer 1: takes rectified festure maps and passes them to the second fully connected layer of 64 neurons.
Fully connected layer 2: passes 64 neurons into 1 sigmoid neuron.
Fully connected layer 3: gives a class probability, 1 is for cats, 0 is for dogs.

## The database
The database that was used for training is a classic Kaggle database for cat and dog images. 600 images went into testing, 9000 for training. 
Database is processed with a help of a DataSet class and then passed into a custom DataLoader.

## Performance 
32 epochs gives 90% accuracy on the testing set. 

## Lessons learned.
Since it's a personal educational project, I'm gonna add some notes for my future self on what to avoid and what to remember:

- don't forget to normalize the data 
- don't forget to initialize the network
- don't forget which Loss function you are using
- don't forget that you are training on cuda
- calculate all of the sizes in your network, otherwise it will never work
- pytorch only works with mini-batches

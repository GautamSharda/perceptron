# about

a perceptron is a fundamental unit of many neural networks

this project provied a simple set of abstractions to experiment with and train perceptrons with different parameters

perceptron.py is intended to be an extremely simple implementation of a perceptron from scratch, without using modern abstractions, for educational purposes

the remainder of this readme is a bit of a mess as i use it to canvas my ideas and results, i hope to clean it up soon

# in progress

parameters: weight data type, number of weights

update_algorithms: sgd, bgd, mgd

# experimenting with

approximated: addition(x, y): x + y

current goal: and(x, y): x*y where x = 0, 1 and y = 0, 1 using thresholding activation function

1 epoch, non-hardcoded differential, random weights, activation function, hookes law data, non-linear functions

floats between 0 and 1 learn better, faster, more stable, no need for clipping etc

experiments confirm can't learn non-linear, loss stops decreasing at some point, but can learn linear

not able to approximate sigmoid still, may be an issue with the partial derivatives calculation, or with data

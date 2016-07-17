# Gradient descent algoritms

http://sebastianruder.com/optimizing-gradient-descent/index.html

## 1. Batch Gradient Descent

Vanilla gradient descent, aka batch gradient descent, computes the gradient of the cost function w.r.t. to the parameters θθ for the entire training datase.

## 2. Stochastic Gradient Descent

Batch gradient descent performs redundant computations for large datasets, as it recomputes gradients for similar examples 
before each parameter update. SGD does away with this redundancy by performing one update at a time. It is therefore usually much faster and can also be used to learn online. 
On the other hand, this ultimately complicates convergence to the exact minimum, as SGD will keep overshooting.

## 2. Mini-batch gradient descent

Mini-batch gradient descent finally takes the best of both worlds and performs an update for every mini-batch of nn training examples

Common mini-batch sizes range between 50 and 256, but can vary for different applications. 
Mini-batch gradient descent is typically the algorithm of choice when training a neural network and the term SGD usually is employed also when mini-batches are used. 

Vanilla mini-batch gradient descent, however, does not guarantee good convergence
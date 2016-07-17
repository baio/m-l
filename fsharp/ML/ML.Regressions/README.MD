# Gradient descent algoritms

http://sebastianruder.com/optimizing-gradient-descent/index.html

## Vanilla gradient descent algorithms

### 1. Batch Gradient Descent

Vanilla gradient descent, aka batch gradient descent, computes the gradient of the cost function w.r.t. to the parameters θθ for the entire training datase.

### 2. Stochastic Gradient Descent

Batch gradient descent performs redundant computations for large datasets, as it recomputes gradients for similar examples 
before each parameter update. SGD does away with this redundancy by performing one update at a time. It is therefore usually much faster and can also be used to learn online. 
On the other hand, this ultimately complicates convergence to the exact minimum, as SGD will keep overshooting.

### 2. Mini-batch gradient descent

Mini-batch gradient descent finally takes the best of both worlds and performs an update for every mini-batch of nn training examples

Common mini-batch sizes range between 50 and 256, but can vary for different applications. 
Mini-batch gradient descent is typically the algorithm of choice when training a neural network and the term SGD usually is employed also when mini-batches are used. 

Vanilla mini-batch gradient descent, however, does not guarantee good convergence

## Gradient descent optimization algorithms

### 1. Nesterov accelerated gradient

Essentially, when using momentum, we push a ball down a hill. The ball accumulates momentum as it rolls downhill, becoming faster and faster on the way (until it reaches its terminal velocity if there is air resistance, i.e. γ<1γ<1). 
However, a ball that rolls down a hill, blindly following the slope, is highly unsatisfactory. We'd like to have a smarter ball, a ball that has a notion of where it is going so that it knows to slow down before the hill slopes up again.
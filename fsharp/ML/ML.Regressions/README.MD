# Gradient descent algorithms (F# implementation)

http://sebastianruder.com/optimizing-gradient-descent/index.html

## Vanilla gradient descent algorithms

### 1. Batch Gradient Descent

Vanilla gradient descent, aka batch gradient descent, computes gradient of the cost function w.r.t.
to the parameters θ for the entire training dataset.

### 2. Stochastic Gradient Descent

Batch gradient descent performs redundant computations for large datasets, as it recomputes gradients for similar examples
before each parameter update. SGD does away with this redundancy by performing one update at a time. It is therefore usually much faster and can also be used to learn online.
On the other hand, this ultimately complicates convergence to the exact minimum, as SGD will keep overshooting.

### 3. Mini-batch gradient descent (SGD)

Mini-batch gradient descent finally takes the best of both worlds and performs an update for every mini-batch of nn training examples

Common mini-batch sizes range between 50 and 256, but can vary for different applications.
Mini-batch gradient descent is typically the algorithm of choice when training a neural network and the term SGD usually is employed also when mini-batches are used.

Vanilla mini-batch gradient descent, however, does not guarantee good convergence

## Gradient descent optimization algorithms

### 1. Nesterov accelerated gradient (NAG)

Essentially, when using momentum, we push a ball down a hill. The ball accumulates momentum as it rolls downhill, becoming faster and faster on the way (until it reaches its terminal velocity if there is air resistance, i.e. γ<1γ<1).
However, a ball that rolls down a hill, blindly following the slope, is highly unsatisfactory. We'd like to have a smarter ball, a ball that has a notion of where it is going so that it knows to slow down before the hill slopes up again.

### 2. Adagrad

It adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters. For this reason, it is well-suited for dealing with sparse data.

### 3. Adadelta

Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate.
Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to some fixed size w.

## Implementation

Every iterative gradient descent algorithm require three principal parameters sets which are independent
one of another.

+ Model parameters
+ Particular Gradient Descent Algorithm parameters
+ Genaral Iterative Model parameters

### 1. Model parameters

Here you define which model you are calculating.
Every gradient descent model requires [gradient descent function]()
in order to minimize [cost function]().

So all `model parameters` require `gradient descent function`.
Also they require `cost function` iteslf in order to benchmark model calculation results for each iteration.

Library includes following models

1. [Linear](http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/)
2. [Logistic](http://ufldl.stanford.edu/tutorial/supervised/LogisticRegression/)
3. [Softmax](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/)

To calculate Softmax model `number of classes` parameters also required.

### 2. Particular Gradient Descent Algorithm parameters

Library includes following algorithms

1. MiniBatch (SGD)
2. Nesterov accelerated gradient (NAG)
3. Adagrad
4. Adadelta

[*SGD*](http://sebastianruder.com/optimizing-gradient-descent/index.html#minibatchgradientdescent) requires following parameters:
+ Alpha - Learning rate
+ BatchSize - Mini Batch size

`SGD` could be converged to
+ `Batch Gradient Descent` when `BatchSize = samples count`
+ `Stochastic Gradient Descent` when `BatchSize = 1`

`SGD` is the base for all other models in the sence all models is calculated using mini batches,
so `BatchSize` parameter required for all models.

Source : `ML.Regressions/SGD.fs`

[*NAG*](http://sebastianruder.com/optimizing-gradient-descent/index.html#nesterovacceleratedgradient) requires following parameters:
+ Alpha - Learning rate
+ BatchSize - Mini Batch size
+ Gamma - momentum term

Source : `ML.Regressions/NAG.fs`

[*Adagrad*](http://sebastianruder.com/optimizing-gradient-descent/index.html#adagrad) requires following parameters:
+ Alpha - Learning rate
+ BatchSize - Mini Batch size (default 0.9)
+ Epsilon - Smoothing term (default 1E-8)

Source : `ML.Regressions/Adagrad.fs`

[*Adadelta*](http://sebastianruder.com/optimizing-gradient-descent/index.html#adadelta) requires following parameters:
+ BatchSize - Mini Batch size
+ Epsilon - Smoothing term (default 1E-8)
+ Rho - momentum term for averaging (default 0.9)

Source : `ML.Regressions/Adadelta.fs`

### 3. General Iterative Model parameters
Here you choose how you iterative model behave
+ Set number of iterations
+ Set convergence flags

### API

`gradientDescent`

Parameters :

+ model : `GLMModel`

See `Model parameters`

```F#
// Linear or Logistic
type GLMBaseModel = {
    Cost : CostFunc
    Gradient : GradientFunc
}

// Softmax
type GLMSoftmaxModel = {
    Base : GLMBaseModel
    ClassesNumber : int
}

type GLMModel =
    | GLMBaseModel of GLMBaseModel
    | GLMSoftmaxModel of GLMSoftmaxModel

```

+ prms: `IterativeTrainModelParams`

See `General Iterative Model parameters`

``` f#
type ConvergeMode =
    //calculate till number of epochs achieved
    | ConvergeModeNone
    //stop calculations when cost function value stops to increase or decrease for 2 consequential iterations
    | ConvergeModeCostStopsChange

type IterativeTrainModelParams = {
    EpochNumber : int
    ConvergeMode : ConvergeMode
}
```

+ inputs: `float Matrix`

Samples features values

+ outputs:  `float Vector`

Samples outputs (labels)

+ hyper : GradientDescentHyperParams

See `Particular Gradient Descent Algorithm parameters`

``` f#

type SGDHyperParams = {
    Alpha: float
    BatchSize: int
}

type NAGHyperParams = {
    Alpha: float
    BatchSize: int
    Gamma: float // momentum term
}

type AdagradHyperParams = {
    Alpha: float
    BatchSize: int
    Epsilon: float
}

type AdadeltaHyperParams = {
    BatchSize: int
    Epsilon: float
    Rho: float
}

type GradientDescentHyperParams =
    | SGDHyperParams of SGDHyperParams
    | NAGHyperParams of NAGHyperParams
    | AdagradHyperParams of AdagradHyperParams
    | AdadeltaHyperParams of AdadeltaHyperParams


```

Source : `ML.Regressions/GradientDescent.fs`

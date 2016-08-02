# Gradient Descent Algorithms Benchmarking

## Linear

```
data:
machine-learning-ex1\ex1\ex1data2.csv
EpochNumber = 5000 // Epochs number
BatchSize = 5
```

### Results
```
===
batch result :
 Theta = seq [340412.6596; 110631.0503; -6649.474246];
 Errors = 2043280051.0;
 CpuTime = 00:00:00.3437500;
===
stochastic result :
 Theta = seq [340412.6596; 110631.0503; -6649.474271];
 Errors = 2043280051.0;
 CpuTime = 00:00:01.1406250;
===
miniBatch result :
 Theta = seq [340412.6596; 110631.0503; -6649.474271];
 Errors = 2043280051.0;
 CpuTime = 00:00:00.3750000;
===
NAG result : {
 Theta = seq [340412.6596; 110631.0503; -6649.474271];
 Errors = 2043280051.0;
 CpuTime = 00:00:00.3750000;
===
Adagrad result :
 Theta = seq [34041264.97; 10463292.12; 5412366.356];
 Errors = 6.6544129e+14;
 CpuTime = 00:00:00.4531250;
===
Adadelta result : {
 Theta = seq [471.3843123; 470.9177804; 470.3712617];
 Errors = 6.53568707e+10;
 CpuTime = 00:00:00.4531250;
```
![linear](./images/linear.png)

### Matlab

```
file: machine-learning-ex1\ex1\ex1_multi.m

Error:  2.0433 * 1.0e+09

Theta computed from gradient descent:
 340412.659574
 110631.050254
 -6649.474246
```

## Logistic

```
data:
machine-learning-ex2\ex2\ex2data1.txt
EpochNumber = 400 // Epochs number
BatchSize = 5
```

### Results
```
batch result :
 Theta = seq [0.2655236719; 0.7466021176; 0.667564853];
 Errors =  0.4106180703;
 CpuTime = 00:00:00.4375000;
===
stochastic result :
 Theta = seq [1.657147293; 3.881418549; 3.616688329];
 Errors = 0.2036090088;
 CpuTime = 00:00:16.6875000;
===
miniBatch result :
 Theta = seq [1.183114387; 2.877677255; 2.647746273];
 Errors = 0.2141232661;
 CpuTime = 00:00:03.4062500;
===
NAG result : {ResultType = EpochCountAchieved;
 Theta = seq [1.183114387; 2.877677255; 2.647746273];
 Errors = 0.2141232661;
 CpuTime = 00:00:03.3750000;
===
Adagrad result : {ResultType = NaN; }
===
Adadelta result : {ResultType = Converged;
 Theta = seq [1.718449394; 4.012902335; 3.743902863];
 Errors =  0.2034977016;
 CpuTime = 00:00:02.4375000;
 GcDelta = [58; 0; 0];}
```
![logistic](./images/logistic.png)

### Matlab

```
file: machine-learning-ex2\ex2\ex2.m

With normalization:

Cost at theta found by fminunc: 0.203498
theta:
 1.718447
 4.012899
 3.743847

Without normalization:

Cost at theta found by fminunc: 0.203498
theta:
 -25.161272
 0.206233
 0.201470

Accuracy: 89.000000
```
> Tip
> Please notice when, do normalisation, values of the thetas will differ from one without normalisation

Interestingly, in this case I couldn't find hyper parameters which works with not normalised values.

Looks like, in this case, octave `fminunc` use Adadelta or similar algorithm.


## Softmax

```
data:
data/iris.csv
EpochNumber = 400 // Epochs number
BatchSize = 5
```

### Result

```
batch result :
 Theta = seq [ -0.1075292102; -0.4148339599; 0.440549417; -0.5967034583; ...];
 Errors = 0.4289583268;
 CpuTime = 00:00:01.0156250;
===
stochastic result :
 Theta = seq [-0.4829995476; -2.033103842; 2.016787211; -3.609455131; ...];
 Errors = 0.06135400897;
 CpuTime = 00:00:53.5937500;
===
miniBatch result :
 Theta = seq [-0.4200879248; -0.8605574447; 0.938587267; -1.353020664; ...];
 Errors = 0.2159036458;
 CpuTime = 00:00:03.0156250;
===
NAG result :
 Theta = seq [-0.4200879248; -0.8605574447; 0.938587267; -1.353020664; ...];
 Errors = 0.2159036458;
 CpuTime = 00:00:03.5156250;
===
Adagrad result :
 Theta = seq [-5.496216329; -28.48130961; 34.56551861; -40.9804075; ...];
 Errors = 0.0396638675;
 CpuTime = 00:00:03.1093750;
===
Adadelta result :
 Theta = seq [-0.8072810257; -3.342523972; 2.455895347; -3.957274757; ...];
 Errors = 0.05675504729;
 CpuTime = 00:00:03.2031250;

```

![softmax](./images/softmax.png)

> accuracy / error / iters

### Tensorfolw

```
Epoch: 0397 cost= 0.231241720
Epoch: 0398 cost= 0.232853398
Epoch: 0399 cost= 0.247151194
Epoch: 0400 cost= 0.227870924
Optimization Finished!
Accuracy: 0.966667
```
Looks like `Tensorflow` use in this case simple `SGD`
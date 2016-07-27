module ML.Regressions.Adagrad

//https://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/

open MathNet.Numerics.LinearAlgebra
open ML.Core.Utils
open ML.Core.LinearAlgebra
open GLM
open GD
open SGD

type AdagradHyperParams = {
    Alpha: float
    BatchSize: int
    Epsilon: float
}

type private AdagradIter = {
    G: float Vector
}

let private calcGradient (prms: CalcGradientParams<AdagradHyperParams>) (iter: GradientDescentIter<AdagradIter>) =
        
    let epsilon = prms.HyperParams.Epsilon
    let alpha = prms.HyperParams.Alpha
    let theta = iter.Theta
    let g = iter.Params.G

    ///
    
    let gradients = prms.Gradient theta prms.X prms.Y
    let k = alpha / (epsilon + g.PointwisePower(0.5))                    
    
    ///

    let updatedTheta = theta - k .* gradients                
    let updatedG = g + gradients.PointwisePower(2.)

    { Theta  = updatedTheta ; Params = { G = updatedG } }
    
let private calcGradient2 (prms: CalcGradientParams<AdagradHyperParams>) (iter: GradientDescentIter<AdagradIter>) =
    calcGradientBatch prms.HyperParams.BatchSize prms iter calcGradient

let private initIter (initialTheta: float Vector) = { Theta  = initialTheta; Params = { G = initialTheta } }
    
let adagrad : GradientDescentFunc<AdagradHyperParams> = 
    GD<AdagradIter, AdagradHyperParams> initIter calcGradient2


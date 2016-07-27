﻿module ML.Regressions.Adagrad

//https://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/

open ML.Core.Utils
open ML.Core.LinearAlgebra
open ML.Regressions.GLM
open MathNet.Numerics.LinearAlgebra
open SGD
open GradientDescent 

type private AdagradIter = {
    Momentum: float Vector
    G: float Vector
}

let private calcGradient (prms: CalcGradientParams<AdagradHyperParams>) (iter: GradientDescentIter<AdagradIter>) =
    let epsilon = prms.HyperParams.Epsilon
    let alpha = prms.HyperParams.NAG.SGD.Basic.Alpha

    let theta = iter.Theta
    let g = iter.Params.G
    
    let gradients = prms.Gradient theta prms.X prms.Y
    let k = alpha / (epsilon + g.PointwisePower(0.5))                    
    
    let updatedTheta = theta - k .* gradients                
    let updatedG = g + gradients.PointwisePower(2.)

    { Theta  = updatedTheta ; Params = { Momentum = iter.Params.Momentum; G = updatedG } }
    
let private calcGradient2 (prms: CalcGradientParams<AdagradHyperParams>) (iter: GradientDescentIter<AdagradIter>) =
    calcGradientBatch prms.HyperParams.NAG.SGD.BatchSize prms iter calcGradient

let private initIter (initialTheta: float Vector) = { Theta  = initialTheta; Params = { Momentum = initialTheta; G = initialTheta } }
    
let adagrad : GradientDescentFunc<AdagradHyperParams> = 
    gradientDescent<AdagradIter, AdagradHyperParams> initIter calcGradient2


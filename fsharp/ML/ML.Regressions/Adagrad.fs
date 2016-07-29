﻿module ML.Regressions.Adagrad

//https://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/

open MathNet.Numerics.LinearAlgebra
open ML.Core.Utils
open ML.Core.LinearAlgebra
open GLM
open GD
open SGD
open Theta

type AdagradHyperParams = {
    Alpha: float
    BatchSize: int
    Epsilon: float
}

type AdagradIter = {
    G: float Vector
}

let private calcGradient (prms: CalcGradientParams<AdagradHyperParams>) (iter: GradientDescentIter<AdagradIter>) =
        
    let epsilon = prms.HyperParams.Epsilon
    let alpha = prms.HyperParams.Alpha
    let g = iter.Params.G
    let theta = iter.Theta    

    //calc grad
    let grad = iter.Theta |> prms.Gradient prms.X prms.Y
    
    //calc grads coefficents
    let k = alpha / (g + epsilon).PointwisePower(0.5)                           
    let updatedTheta = theta - k .* grad
    
    //accumulate G
    let updatedG = (g + grad).PointwisePower(2.)

    { Theta  = updatedTheta ; Params = { G = updatedG } }
    
let private calcGradient2 (prms: CalcGradientParams<AdagradHyperParams>) (iter: GradientDescentIter<AdagradIter>) =
    calcGradientBatch prms.HyperParams.BatchSize prms iter calcGradient

    
let adagrad     
    (model: GLMModel)
    (prms: IterativeTrainModelParams)    
    (hyperPrms : AdagradHyperParams)
    (x : float Matrix)
    (y : float Vector) 
    =
    let shape, theta, baseModel = getModelShapeAndTheta model x.ColumnCount    
    GD<AdagradIter, AdagradHyperParams> calcGradient2 shape {Theta = theta; Params = { G = theta }} baseModel prms hyperPrms x y              
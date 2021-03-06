﻿// http://arxiv.org/pdf/1212.5701v1.pdf
// http://climin.readthedocs.io/en/latest/adadelta.html
module ML.GD.Adadelta

open MathNet.Numerics.LinearAlgebra
open ML.Core.Utils
open ML.Core.LinearAlgebra
open GLM
open GD
open SGD

type AdadeltaHyperParams = {
    BatchSize: int
    Epsilon: float
    Rho: float
}

type AdadeltaIter = {
    EG: float Vector
    ET: float Vector
}

let private calcGradient (prms: CalcGradientParams<AdadeltaHyperParams>) (iter: GradientDescentIter<AdadeltaIter>) =
        
    let epsilon = prms.HyperParams.Epsilon    
    let rho = prms.HyperParams.Rho
    let eg = iter.Params.EG
    let et = iter.Params.ET
    let theta = iter.Theta

    //calculate gradient  
    let grad = iter.Theta |> prms.Gradient prms.X prms.Y
    //accumulate gradient
    let updatedEG = grad.PointwisePower(2.) * (1. - rho) + eg * rho                   
    //compute update
    let rms_t = (et + epsilon).PointwisePower(0.5)
    let rms_g = (updatedEG + epsilon).PointwisePower(0.5)
    let delta = grad .* (rms_t / rms_g) 
    //accumulate updates
    let updatedET = et * rho + delta.PointwisePower(2.) * (1. - rho)                   
    //apply update
    let updatedTheta = theta - delta

    { Theta  = updatedTheta; Params = { EG = updatedEG; ET = updatedET } }
    
let private calcGradient2
    iterParamsUpdate
    (prms: CalcGradientParams<AdadeltaHyperParams>) 
    (iter: GradientDescentIter<AdadeltaIter>) 
    =
    calcGradientBatch iterParamsUpdate prms.HyperParams.BatchSize prms iter calcGradient
    
let adadelta
    (iterParamsProvider: IterParamsProvider<AdadeltaIter>)
    (model: GLMModel)
    (prms: IterativeTrainModelParams)    
    (hyperPrms : AdadeltaHyperParams)
    (x : float Matrix)
    (y : float Vector) 
    =
    let shape, _, baseModel = getModelShapeAndTheta model x.ColumnCount   
    
   
    GD<AdadeltaIter, AdadeltaHyperParams> (calcGradient2 iterParamsProvider.update) shape (iterParamsProvider.initial()) baseModel prms hyperPrms x y              

//{Theta = theta; Params = { EG = theta; ET = theta }}
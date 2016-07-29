// http://arxiv.org/pdf/1212.5701v1.pdf
// http://climin.readthedocs.io/en/latest/adadelta.html
module ML.Regressions.Adadelta

open MathNet.Numerics.LinearAlgebra
open ML.Core.Utils
open ML.Core.LinearAlgebra
open GLM
open GD
open SGD
open Theta

type AdadeltaHyperParams = {
    BatchSize: int
    Epsilon: float
    Rho: float
}

type AdadeltaIter = {
    EG: Theta
    ET: Theta
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
    let updatedEG = ( grad .^ 2.) * (1. - rho) + eg * rho                   
    //compute update
    let rms_t = (et + epsilon) .^ 0.5
    let rms_g = (updatedEG + epsilon) .^  0.5
    let delta = grad .* (rms_t / rms_g) 
    //accumulate updates
    let updatedET = et * rho + (delta .^ 2.) * (1. - rho)                   
    //apply update
    let updatedTheta = theta - delta

    { Theta  = updatedTheta; Params = { EG = updatedEG; ET = updatedET } }
    
let private calcGradient2 (prms: CalcGradientParams<AdadeltaHyperParams>) (iter: GradientDescentIter<AdadeltaIter>) =
    calcGradientBatch prms.HyperParams.BatchSize prms iter calcGradient

let private initIter (initialTheta: float Vector) = 
    let theta = ThetaVector(initialTheta)
    { Theta  = theta; Params = { EG = theta; ET = theta } }
    
let adadelta (initialIter : GradientDescentIter<AdadeltaIter>) : GradientDescentFunc<AdadeltaHyperParams> = 
    GD<AdadeltaIter, AdadeltaHyperParams> calcGradient2 initialIter

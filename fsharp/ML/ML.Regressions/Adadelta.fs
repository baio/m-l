// http://arxiv.org/pdf/1212.5701v1.pdf
// http://climin.readthedocs.io/en/latest/adadelta.html
module ML.Regressions.Adadelta

open ML.Core.Utils
open ML.Core.LinearAlgebra
open ML.Regressions.GLM
open MathNet.Numerics.LinearAlgebra
open SGD
open GradientDescent 

type private AdadeltaIter = {
    EG: float Vector
    ET: float Vector 
}

let private calcGradient (prms: CalcGradientParams<AdadeltaHyperParams>) (iter: GradientDescentIter<AdadeltaIter>) =
        
    let epsilon = prms.HyperParams.Epsilon
    let alpha = prms.HyperParams.SGD.Basic.Alpha
    let theta = iter.Theta
    let rho = prms.HyperParams.Rho
    let eg = iter.Params.EG
    let et = iter.Params.ET

    //calculate gradient  
    let gradients = prms.Gradient theta prms.X prms.Y
    //accumulate gradient
    let updatedEG = rho * eg + (1. - rho) * gradients.PointwisePower(2.)                    
    //compute update
    let rms_t = (et + epsilon).PointwisePower(0.5)
    let rms_g = (updatedEG + epsilon).PointwisePower(0.5)
    let delta = (rms_t / rms_g) .* gradients
    //accumulate updates
    let updatedET = rho * et + (1. - rho) * delta.PointwisePower(2.)                    
    //apply update
    let updatedTheta = theta - delta

    { Theta  = updatedTheta ; Params = { EG = updatedEG; ET = updatedET } }
    
let private calcGradient2 (prms: CalcGradientParams<AdadeltaHyperParams>) (iter: GradientDescentIter<AdadeltaIter>) =
    calcGradientBatch prms.HyperParams.SGD.BatchSize prms iter calcGradient

let private initIter (initialTheta: float Vector) = { Theta  = initialTheta; Params = { EG = initialTheta; ET = initialTheta } }
    
let adadelta : GradientDescentFunc<AdadeltaHyperParams> = 
    gradientDescent<AdadeltaIter, AdadeltaHyperParams> initIter calcGradient2

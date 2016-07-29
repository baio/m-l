module ML.Regressions.NAG
//http://stats.stackexchange.com/questions/179915/whats-the-difference-between-momentum-based-gradient-descent-and-nesterovs-ac

open MathNet.Numerics.LinearAlgebra
open ML.Core.Utils
open ML.Core.LinearAlgebra
open ML.Regressions.GLM
open SGD
open GD
open Theta

type NAGHyperParams = {
    Alpha: float
    BatchSize: int
    Gamma: float // momentum term
}

type NAGIter = {
    Momentum: Theta
}

let calcGradient (prms: CalcGradientParams<NAGHyperParams>) (iter: GradientDescentIter<NAGIter>) =
    let alpha = prms.HyperParams.Alpha
    let theta = iter.Theta
    let momentum = iter.Params.Momentum 
    let a = momentum * prms.HyperParams.Gamma
    let grad = iter.Theta - a |> prms.Gradient prms.X prms.Y 
    let updatedMomentum = grad * alpha + a
    
    { Theta  = theta - updatedMomentum; Params = { Momentum = momentum } }
    
let private calcGradient2 (prms: CalcGradientParams<NAGHyperParams>) (iter: GradientDescentIter<NAGIter>) =
    calcGradientBatch prms.HyperParams.BatchSize prms iter calcGradient

let private initIter (initialTheta: float Vector) = 
    let theta = ThetaVector(initialTheta)
    { Theta  = theta; Params = { Momentum = theta } }
    
let NAG : GradientDescentFunc<NAGHyperParams> = 
    GD<NAGIter, NAGHyperParams> calcGradient2 initIter

module ML.Regressions.NAG
//http://stats.stackexchange.com/questions/179915/whats-the-difference-between-momentum-based-gradient-descent-and-nesterovs-ac

open ML.Core.Utils
open ML.Core.LinearAlgebra
open ML.Regressions.GLM
open MathNet.Numerics.LinearAlgebra
open SGD
open GradientDescent 

type private NAGIter = {
    Momentum: float Vector
}

let private calcGradient (prms: CalcGradientParams<NAGHyperParams>) (iter: GradientDescentIter<NAGIter>) =
    let theta = iter.Theta

    let alpha = prms.HyperParams.SGD.Basic.Alpha
    let a = prms.HyperParams.Gamma * iter.Params.Momentum
    let gradients = prms.Gradient (theta - a) prms.X prms.Y
    let momentum = a + prms.HyperParams.SGD.Basic.Alpha * gradients

    { Theta  = theta - alpha * gradients; Params = { Momentum = momentum } }
    
let private calcGradient2 (prms: CalcGradientParams<NAGHyperParams>) (iter: GradientDescentIter<NAGIter>) =
    calcGradientBatch prms.HyperParams.SGD.BatchSize prms iter calcGradient

let private initIter (initialTheta: float Vector) = { Theta  = initialTheta; Params = { Momentum = initialTheta } }
    
let NAG : GradientDescentFunc<NAGHyperParams> = 
    gradientDescent<NAGIter, NAGHyperParams> initIter calcGradient2

module ML.Regressions.NAG
//http://stats.stackexchange.com/questions/179915/whats-the-difference-between-momentum-based-gradient-descent-and-nesterovs-ac

open MathNet.Numerics.LinearAlgebra
open ML.Core.Utils
open ML.Core.LinearAlgebra
open ML.Regressions.GLM
open SGD
open GD

type NAGHyperParams = {
    Alpha: float
    BatchSize: int
    Gamma: float // momentum term
}

type NAGIter = {
    Momentum: float Vector
}

let calcGradient (prms: CalcGradientParams<NAGHyperParams>) (iter: GradientDescentIter<NAGIter>) =
    let alpha = prms.HyperParams.Alpha
    let theta = iter.Theta
    let momentum = iter.Params.Momentum 
    let a = momentum * prms.HyperParams.Gamma
    let grad = theta - a |> prms.Gradient prms.X prms.Y 
    let updatedMomentum = grad * alpha + a
    
    { Theta  = theta - updatedMomentum; Params = { Momentum = momentum } }
    
let private calcGradient2 
    (iterParamsUpdate)
    (prms: CalcGradientParams<NAGHyperParams>) 
    (iter: GradientDescentIter<NAGIter>) =
    calcGradientBatch iterParamsUpdate prms.HyperParams.BatchSize prms iter calcGradient
    
let NAG2
    (iterParamsUpdate) 
    (model: GLMModel)
    (prms: IterativeTrainModelParams)    
    (hyperPrms : NAGHyperParams)
    (x : float Matrix)
    (y : float Vector) 
    =         
    let shape, theta, baseModel = getModelShapeAndTheta model x.ColumnCount    
    GD<NAGIter, NAGHyperParams> (calcGradient2 iterParamsUpdate) shape { Theta = theta ; Params = { Momentum = theta } } baseModel prms hyperPrms x y              

let NAG : GradientDescentFunc<NAGHyperParams> = NAG2 (fun p -> p)
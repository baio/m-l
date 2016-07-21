module ML.Regressions.GLM

open ML.Core.Utils
open ML.Core.LinearAlgebra
open MathNet.Numerics.LinearAlgebra
// GLM 
// http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/

// Given weights and features return calculated label
type HypothesisFunc = float Vector -> float Vector -> float
// Given weights, features and labels calculate error
type LossFunc = float Vector -> float Matrix -> float Vector -> float
// Given weights, inputs and outputs calculate gradient array for weights
type GradientFunc = float Vector -> float Matrix -> float Vector -> float Vector
// Given HypothesisFunc returns cost and gradient functions
type GenLossAndGradientFunc = HypothesisFunc -> LossFunc * GradientFunc

type GLMModel = {
    Hypothesis : HypothesisFunc
    Loss : LossFunc   
    Gradient : GradientFunc   
}

type AcceleratedTrainModelParams = {
    EpochNumber : int // number of training iterations
    Alpha: float // learning rate
    BatchSize: int // mini batch size
    Gamma: float // momentum term
    MinErrorThreshold : float
}

type MiniBatchTrainModelParams = {
    MaxIterNumber : int
    MinErrorThreshold : float
    Alpha: float
    BatchSize: int
}

type IterativeTrainModelParams = {
    MaxIterNumber : int
    MinErrorThreshold : float
    Alpha: float
}


type ModelTrainResult = Converged | ErrorThresholdAchieved | MaxIterCountAchieved


let predict (w: float Vector) (x: float Vector) = 
    x |> vecCons 1. |> (*) w

let predictNorm (normPrms: NormParams) (w: float Vector) (x: float Vector) =     
    (x - normPrms.Mu) ./ normPrms.Std |> predict w

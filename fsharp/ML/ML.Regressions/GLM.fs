﻿module ML.Regressions.GLM

open ML.Core.Utils
open ML.Core.LinearAlgebra
open MathNet.Numerics.LinearAlgebra
// GLM 
// http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/

// Given weights and features return calculated label
type HypothesisFunc = float Vector -> float Vector -> float
// Given weights, features and labels calculate error
type CostFunc = float Vector -> float Matrix -> float Vector -> float
// Given weights, inputs and outputs calculate gradient array for weights
type GradientFunc = float Vector -> float Matrix -> float Vector -> float Vector
// Given HypothesisFunc returns cost and gradient functions
type GenLossAndGradientFunc = HypothesisFunc -> CostFunc * GradientFunc

type GLMModel = {    
    Hypothesis : HypothesisFunc
    Cost : CostFunc   
    Gradient : GradientFunc   
}

type BasicHyperParams = {
    Alpha: float
}

type SGDHyperParams = {
    Basic: BasicHyperParams
    BatchSize: int
}

type NAGHyperParams = {
    SGD: SGDHyperParams
    Gamma: float // momentum term
}

type AdagradHyperParams = {
    SGD: SGDHyperParams
    Epsilon: float
}

type AdadeltaHyperParams = {
    SGD: SGDHyperParams
    Epsilon: float
    Rho: float
}

type RegressionHyperParams = 
    | BasicHyperParams of BasicHyperParams
    | SGDHyperParams of SGDHyperParams
    | NAGHyperParams of NAGHyperParams
    | AdagradHyperParams of AdagradHyperParams
    | AdadeltaHyperParams of AdadeltaHyperParams 
     
type ConvergeMode = 
    | ConvergeModeNone
    | ConvergeModeCostStopsChange

type IterativeTrainModelParams = {
    EpochNumber : int
    ConvergeMode : ConvergeMode
}

type ModelTrainResultType = Converged | ErrorThresholdAchieved | MaxIterCountAchieved

type ModelTrainResult = { ResultType : ModelTrainResultType; Weights: float Vector; Errors: float list }

type WeightsCalculator = float Vector -> float Matrix -> float Vector -> float Vector

let predict (w: float Vector) (x: float Vector) = 
    x |> vecCons 1. |> (*) w

let predictNorm (normPrms: NormParams) (w: float Vector) (x: float Vector) =     
    (x - normPrms.Mu) ./ normPrms.Std |> predict w


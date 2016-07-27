module ML.Regressions.GLM

open ML.Core.Utils
open ML.Core.LinearAlgebra
open MathNet.Numerics.LinearAlgebra

// Given weights and features return calculated label
type HypothesisFunc = float Vector -> float Vector -> float
// Given weights, features and labels calculate error
type CostFunc = float Vector -> float Matrix -> float Vector -> float
// Given weights, inputs and outputs calculate gradient array for weights
type GradientFunc = float Vector -> float Matrix -> float Vector -> float Vector

type GLMModel = {    
    Hypothesis : HypothesisFunc
    Cost : CostFunc   
    Gradient : GradientFunc   
}
    
let predict (theta: float Vector) (x: float Vector) = 
    x |> vecCons 1. |> (*) theta

let predictNorm (normPrms: NormParams) (theta: float Vector) (x: float Vector) =     
    (x - normPrms.Mu) ./ normPrms.Std |> predict theta


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
    Cost : CostFunc   
    Gradient : GradientFunc   
}
    
let GLMPredict (hypothesis: HypothesisFunc) (theta: float Vector) (x: float Vector) = 
    hypothesis (x |> vecCons 1.) theta

let GLMPredictNorm (hypothesis: HypothesisFunc) (normPrms: NormParams) (theta: float Vector) (x: float Vector) =     
    (x - normPrms.Mu) ./ normPrms.Std |> GLMPredict hypothesis theta


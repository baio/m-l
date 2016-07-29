module ML.Regressions.GLM

open ML.Core.Utils
open ML.Core.LinearAlgebra
open MathNet.Numerics.LinearAlgebra

open Theta

type ThetaShape = 
    | ThetaShapeVector
    | ThetaShapeMatrix of int * int
    member this.matrixSize() = 
        match this with 
        | ThetaShapeMatrix (r, c) -> (r, c)
        | _ -> failwith "Shape is not a matrix"

// Given weights and features return calculated label
type HypothesisFunc = float Vector -> Theta -> float
// Given features and labels, weights calculate error
type CostFunc = ThetaShape -> float Matrix -> float Vector -> float Vector -> float
// Given inputs and outputs, weights calculate gradient array for weights
type GradientFunc = ThetaShape -> float Matrix -> float Vector -> float Vector -> float Vector


type GLMModel = {       
    Cost : CostFunc
    Gradient : GradientFunc
}

(*    
let GLMPredict (hypothesis: HypothesisFunc) (theta: float Vector) (x: float Vector) = 
    hypothesis (x |> vecCons 1.) theta

let GLMPredictNorm (hypothesis: HypothesisFunc) (normPrms: NormParams) (theta: float Vector) (x: float Vector) =     
    (x - normPrms.Mu) ./ normPrms.Std |> GLMPredict hypothesis theta
*)


module ML.Regressions.LogisticRegression

open MathNet.Numerics.LinearAlgebra
open ML.Core.Utils
open ML.Core.LinearAlgebra

open GLM

let sigmoid (z: float Vector) = 
    1. / (1. + (-1. * z).PointwiseExp())

let sigmoidScalar (z: float) = 
    1. / (1. + System.Math.Exp(-1. * z))

let logisticHyp (x: float Vector) (theta: float Vector) = 
     theta * x |> sigmoidScalar |> System.Math.Round

let logisticMSECost _ (x : float Matrix) (y : float Vector) (theta: float Vector) =
    let s = x *  theta |> sigmoid
    let p = y * s.PointwiseLog()     
    let n = (1. - y) * (1. - s).PointwiseLog()
    - 1. * ( p + n ) / float x.RowCount
        
let logisticMSEGradient _ (x : float Matrix) (y : float Vector) (theta: float Vector) =
    x.Transpose() * (sigmoid(x * theta) - y) / float x.RowCount

(*
let predict : float Vector -> float Vector -> float = GLMPredict logisticHyp

let predictNorm : NormParams -> float Vector -> float Vector -> float = GLMPredictNorm logisticHyp
*)

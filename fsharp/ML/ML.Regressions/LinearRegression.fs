module ML.Regressions.LinearRegression

open ML.Core.Utils
open MathNet.Numerics.LinearAlgebra

open GLM
open Theta

let linearMSECost (x : float Matrix) (y : float Vector) (theta: Theta) =         
    ((x * theta.asVector() - y).PointwisePower(2.) |> Vector.sum) / (2. * float x.RowCount)
            
let linearMSEGradient  (x : float Matrix) (y : float Vector) (theta: Theta) =
    x.Transpose() * (x * theta.asVector() - y) / float x.RowCount 

let linearHyp (x: float Vector) (theta: Theta) = theta.asVector() * x                

(*
let predict : float Vector -> float Vector -> float = GLMPredict linearHyp

let predictNorm : NormParams -> float Vector -> float Vector -> float = GLMPredictNorm linearHyp
*)
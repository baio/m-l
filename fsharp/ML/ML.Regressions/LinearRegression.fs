module ML.Regressions.LinearRegression

open ML.Core.Utils
open MathNet.Numerics.LinearAlgebra

open GLM
open Theta

let linearHyp (x: float Vector) (theta: Theta) = theta.vector() * x                

let linearMSECost _ (x : float Matrix) (y : float Vector) (theta: float Vector)  =     
    ((x * theta - y).PointwisePower(2.) |> Vector.sum) / (2. * float x.RowCount) 
        
let linearMSEGradient _ (x : float Matrix) (y : float Vector) (theta: float Vector) =
    x.Transpose() * (x * theta - y) / float x.RowCount

(*
let predict : float Vector -> float Vector -> float = GLMPredict linearHyp

let predictNorm : NormParams -> float Vector -> float Vector -> float = GLMPredictNorm linearHyp
*)
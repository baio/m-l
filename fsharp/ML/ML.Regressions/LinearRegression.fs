module ML.Regressions.LinearRegression

open ML.Core.Utils
open MathNet.Numerics.LinearAlgebra

open GLM

let linearMSECost (theta: float Vector) (x : float Matrix) (y : float Vector) =     
    ((x * theta - y).PointwisePower(2.) |> Vector.sum) / (2. * float x.RowCount)
        
let linearMSEGradient (theta: float Vector) (x : float Matrix) (y : float Vector) =
    x.Transpose() * (x * theta - y) / float x.RowCount 

let linearHyp (theta: float Vector) (x: float Vector) = theta * x                

let predict : float Vector -> float Vector -> float = GLMPredict linearHyp

let predictNorm : NormParams -> float Vector -> float Vector -> float = GLMPredictNorm linearHyp
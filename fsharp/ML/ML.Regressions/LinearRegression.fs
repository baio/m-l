module ML.Regressions.LinearRegression

open ML.Core.Utils
open MathNet.Numerics.LinearAlgebra

open GLM

let linearMSECost (w: float Vector) (x : float Matrix) (y : float Vector) =     
    ((x * w - y).PointwisePower(2.) |> Vector.sum) / (2. * float x.RowCount)
        
let linearMSEGradient (w: float Vector) (x : float Matrix) (y : float Vector) =
    x.Transpose() * (x * w - y) / float x.RowCount 

let linearHyp (theta: float Vector) (x: float Vector) = theta * x                

let predict : float Vector -> float Vector -> float = GLMPredict linearHyp

let predictNorm : NormParams -> float Vector -> float Vector -> float = GLMPredictNorm linearHyp
module ML.GD.LinearRegression

open ML.Core.Utils
open MathNet.Numerics.LinearAlgebra

open GLM

let linearHyp (x: float Vector) (theta: float Vector) = theta * x                

let linearMSECost _ (x : float Matrix) (y : float Vector) (theta: float Vector)  =     
    ((x * theta - y).PointwisePower(2.) |> Vector.sum) / (2. * float x.RowCount) 
        
let linearMSEGradient _ (x : float Matrix) (y : float Vector) (theta: float Vector) =
    x.Transpose() * (x * theta - y) / float x.RowCount

let predict : float Vector -> float Matrix -> float Vector = GLMPredict linearHyp

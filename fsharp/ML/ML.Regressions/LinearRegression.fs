module ML.Regressions.LinearRegression

open MathNet.Numerics.LinearAlgebra

let linearHyp (theta: float Vector) (x: float Vector) = 
     theta * (x |> Vector.insert 0 1.)

let linearMSECost (w: float Vector) (x : float Matrix) (y : float Vector) =     
    ((x * w - y).PointwisePower(2.) |> Vector.sum) / (2. * float x.RowCount)
        
let linearMSEGradient (w: float Vector) (x : float Matrix) (y : float Vector) =
    x.Transpose() * (x * w - y) / float x.RowCount 
               

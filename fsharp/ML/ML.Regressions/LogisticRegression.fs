module ML.Regressions.LogisticRegression

open MathNet.Numerics.LinearAlgebra
open ML.Core.LinearAlgebra

let sigmoid (z: float Vector) = 
    1. / (1. + (-1. * z).PointwiseExp())

let sigmoidScalar (z: float) = 
    1. / (1. + System.Math.Exp(-1. * z))

let logisticHyp (theta: float Vector) (x: float Vector) = 
     x |> appendOne |> (*) theta |> sigmoidScalar |> System.Math.Round

let logisticMSECost (w: float Vector) (x : float Matrix) (y : float Vector) =
    let s = x * w |> sigmoid
    let p = y * s.PointwiseLog()     
    let n = (1. - y) * (1. - s).PointwiseLog()
    - 1. * ( p + n ) / float x.RowCount
        
let logisticMSEGradient (w: float Vector) (x : float Matrix) (y : float Vector) =
    x.Transpose() * (sigmoid(x * w) - y) / float x.RowCount 

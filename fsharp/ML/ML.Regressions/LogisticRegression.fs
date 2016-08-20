module ML.GD.LogisticRegression

open MathNet.Numerics.LinearAlgebra
open ML.Core.Utils
open ML.Core.LinearAlgebra

open GLM

let sigmoid (z: float Vector) = 
    1. / (1. + (-1. * z).PointwiseExp())

let sigmoid' (z: float Vector) = 
    sigmoid(z) .* (1. - sigmoid(z))

let sigmoidScalar (z: float) = 
    1. / (1. + System.Math.Exp(-1. * z))


let logisticMSECost _ (x : float Matrix) (y : float Vector) (theta: float Vector) =
    let s = x *  theta |> sigmoid
    let p = y * s.PointwiseLog()     
    let n = (1. - y) * (1. - s).PointwiseLog()
    - 1. * ( p + n ) / float x.RowCount
        
let logisticMSEGradient _ (x : float Matrix) (y : float Vector) (theta: float Vector) =
    //printfn "%A" theta
    x.Transpose() * (sigmoid(x * theta) - y) / float x.RowCount    

let logisticHyp (theta: float Vector) (x: float Vector)  = 
     theta * x |> sigmoidScalar |> System.Math.Round

// theta, x, y
let predict : float Vector -> float Matrix -> float Vector = GLMPredict logisticHyp

// theta, x, y
let accuracy : float Vector -> float Matrix -> float Vector -> float = GLMAccuracy logisticHyp

module ML.GD.GLM

open ML.Core.Utils
open ML.Core.LinearAlgebra
open MathNet.Numerics.LinearAlgebra


type ThetaShape = 
    | ThetaShapeVector
    | ThetaShapeMatrix of int * int
    member this.matrixSize() = 
        match this with 
        | ThetaShapeMatrix (r, c) -> (r, c)
        | _ -> failwith "Shape is not a matrix"

// Given weights and features return calculated label
type HypothesisFunc = float Vector -> float Vector -> float
// Given features and labels, weights calculate error
type CostFunc = ThetaShape -> float Matrix -> float Vector -> float Vector -> float
// Given inputs and outputs, weights calculate gradient array for weights
type GradientFunc = ThetaShape -> float Matrix -> float Vector -> float Vector -> float Vector

// Linear or Logistic
type GLMBaseModel = {       
    Cost : CostFunc
    Gradient : GradientFunc
}

// Softmax
type GLMSoftmaxModel = {       
    Base : GLMBaseModel
    ClassesNumber : int
}

type GLMModel = 
    | GLMBaseModel of GLMBaseModel
    | GLMSoftmaxModel of GLMSoftmaxModel


let GLMPredict (hypothesis: HypothesisFunc) (theta: float Vector) (mx: float Matrix) =     
    DenseVector.init mx.RowCount (fun i -> 
        mx.Row(i) |> appendOne |> hypothesis theta 
    )

let GLMCorrectPercent (y: float Vector) (actual: float Vector) =
    let correct = actual |> Vector.mapi (fun i a -> if y.[i] = a then 1. else 0.) |> Vector.sum
    correct / float y.Count 

let GLMAccuracy (hypothesis: HypothesisFunc) (theta: float Vector) (x: float Matrix) (y: float Vector) =
    GLMPredict hypothesis theta x
    |> GLMCorrectPercent y
    
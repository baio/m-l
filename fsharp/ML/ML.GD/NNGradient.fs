module ML.GD.NNGradient

open MathNet.Numerics.LinearAlgebra
open ML.Core.LinearAlgebra
open ML.Core.Utils

open GLM
open ML.NN
open Nessos.Streams

let NNCost (thetaShape: ThetaShape) (x : FMatrix) (y : FVector) (theta: FVector)  =     
    let shape = thetaShape.nnShape()    
    //TODO : forward require inputs without bias
    let x = x.RemoveColumn(0)
    let fwd = forward x shape theta
    let ssum = (mxSubVec fwd  y).PointwisePower(2.) |> Matrix.sum
    ssum / 2. * float x.RowCount
        
let NNGradient (thetaShape: ThetaShape) (x : FMatrix) (y : FVector) (theta: FVector) =
    let x = x.RemoveColumn(0)
    backprop y x (thetaShape.nnShape()) theta: FVector

let predict (mapOutput: FVector -> FVector) (shape: NNShape) (x: float Matrix) (theta: float Vector) : FMatrix =         
    forward x shape theta 
    |> Matrix.mapCols (fun _ c -> mapOutput c)
        
// number of classes, theta, x, y
let accuracy (mapOutput: FVector -> FVector) (shape: NNShape) (x: float Matrix) (y: float Vector) (theta: float Vector) : float = 
    let ys = chunkOutputs x.RowCount y
    let actual = 
        predict mapOutput shape x theta 
    let correct = 
        actual.EnumerateRows()
        |> Seq.zip ys
        |> Seq.map (ifeq 1 0)
        |> Seq.sum

    float correct / float x.RowCount

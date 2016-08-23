module ML.GD.NNGradient

open MathNet.Numerics.LinearAlgebra
open ML.Core.LinearAlgebra
open ML.Core.Utils

open GLM
open ML.NN
open Nessos.Streams

let NNCost (thetaShape: ThetaShape) (x : FMatrix) (y : FVector) (theta: FVector)  =     
    let shape = thetaShape.nnShape()    
    let ys = chunkOutputs x.RowCount y
    //TODO : forward require inputs without bias
    let x = x.RemoveColumn(0)
    let fwd = forward x shape theta
    let ssum = (fwd - ys).PointwisePower(2.).RowSums().Sum()
    ssum / (2. * float x.RowCount)
        
let NNGradient (thetaShape: ThetaShape) (x : FMatrix) (y : FVector) (theta: FVector) =
    let x = x.RemoveColumn(0)
    backprop y x (thetaShape.nnShape()) theta: FVector

let predict (mapOutput: FVector -> FVector) (shape: NNShape) (x: float Matrix) (theta: float Vector) : FMatrix =         
    forward x shape theta 
    |> Matrix.mapRows (fun _ c -> mapOutput c)
        
// number of classes, theta, x, y
let accuracy (mapOutput: FVector -> FVector) (shape: NNShape) (x: float Matrix) (y: float Vector) (theta: float Vector) : float = 
    let ys = chunkOutputs x.RowCount y
    let actual = predict mapOutput shape x theta 
    let correct = 
        ys - actual //diff target - calcualted; [1; 1] - [1; 1] = [0; 0]
        |> Matrix.sumRows // sum diff for each sample, if calculated = 0 then it should be 0.
        |> Vector.map (fun f -> iif (f=0.) 1. 0.)
        |> Vector.sum
    
    float correct / float x.RowCount

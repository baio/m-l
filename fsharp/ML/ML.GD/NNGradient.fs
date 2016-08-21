﻿module ML.GD.NNGradient

open MathNet.Numerics.LinearAlgebra
open ML.Core.LinearAlgebra
open ML.Core.Utils

open GLM
open ML.NN


let NNCost (thetaShape: ThetaShape) (x : FMatrix) (y : FVector) (theta: FVector)  =     
    let shape = thetaShape.nnShape()    
    let ys = chunkOutputs x.RowCount y
    //TODO : forward require inputs without bias
    let x = x.RemoveColumn(0)
    let errSum = 
        x.EnumerateRows() 
        |> Seq.mapi(fun i row ->
            let out = forward row shape theta
            (out - ys.[i]).PointwisePower(2.) |> Vector.sum
        )
        |> Seq.sum
    errSum / (2. * float x.RowCount) 
        
let NNGradient (thetaShape: ThetaShape) (x : FMatrix) (y : FVector) (theta: FVector) =
    let shape = thetaShape.nnShape()
    let ys = chunkOutputs x.RowCount y
    //TODO : forward require inputs without bias
    let x = x.RemoveColumn(0)
    let gradSum = 
        x.EnumerateRows()
        |> Seq.mapi(fun i f ->
            let b = backprop ys.[i] f shape theta
            //System.Diagnostics.Debug.WriteLine(sprintf "%A" b)
            let f = b |> flatNNGradients
            f
        )
        |> DenseMatrix.ofColumnSeq
        |> Matrix.sumRows
    gradSum / float x.RowCount

// number of classes, theta, x, y
let predict (shape: NNShape) (x: float Matrix) (theta: float Vector) : FVector seq =         
    x.EnumerateRows()
    |> Seq.map (fun row -> 
        forward row shape theta
    )
        
// number of classes, theta, x, y
let accuracy (shape: NNShape) (x: float Matrix) (y: float Vector) (theta: float Vector) : float = 
    let ys = chunkOutputs x.RowCount y
    let actual = predict shape x theta |> Seq.map (fun f -> [iif (f.At(0) < 0.5) 0. 1.] |> vector)      
    let correct = 
        actual 
        |> Seq.zip ys
        |> Seq.map (ifeq 1 0)
        |> Seq.sum

    float correct / float x.RowCount

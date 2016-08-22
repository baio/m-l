module ML.GD.NNGradient

open MathNet.Numerics.LinearAlgebra
open ML.Core.LinearAlgebra
open ML.Core.Utils

open GLM
open ML.NN
open Nessos.Streams

let NNCost (thetaShape: ThetaShape) (x : FMatrix) (y : FVector) (theta: FVector)  =     
    let shape = thetaShape.nnShape()    
    let thetasCount = shape.thetasCount()
    let ys = chunkOutputs x.RowCount y
    //TODO : forward require inputs without bias
    let x = x.RemoveColumn(0)
    let errSum = 
        x.EnumerateRows() 
        |> Stream.ofSeq
        |> Stream.mapi(fun i row ->
            let ot = forward row shape theta
            (ot - ys.[i]).PointwisePower(2.) |> Vector.sum
        )
        |> Stream.sum
    errSum / (2. * float x.RowCount) 
        
let NNGradient (thetaShape: ThetaShape) (x : FMatrix) (y : FVector) (theta: FVector) =
    let shape = thetaShape.nnShape()
    let thetasCount = shape.thetasCount()
    let ys = chunkOutputs x.RowCount y
    //TODO : forward require inputs without bias
    // TODO !!! : Stream
    let x = x.RemoveColumn(0)
    let gradSum = 
        x.EnumerateRows()
        |> Stream.ofSeq
        |> Stream.mapi(fun i f ->
            //System.Diagnostics.Debug.WriteLine(sprintf "%A" b)
            backprop ys.[i] f shape theta |> flatMxs
        )
        |> Stream.fold (fun acc v -> acc + v) (zeros thetasCount)
    gradSum / float x.RowCount

// number of classes, theta, x, y
let predict (mapOutput: FVector -> FVector) (shape: NNShape) (x: float Matrix) (theta: float Vector) : FVector seq =         
    x.EnumerateRows()
    |> Seq.map (fun row -> 
        forward row shape theta 
        |> mapOutput
    )
        
// number of classes, theta, x, y
let accuracy (mapOutput: FVector -> FVector) (shape: NNShape) (x: float Matrix) (y: float Vector) (theta: float Vector) : float = 
    let ys = chunkOutputs x.RowCount y
    let actual = 
        predict mapOutput shape x theta 
    let correct = 
        actual 
        |> Seq.zip ys
        |> Seq.map (ifeq 1 0)
        |> Seq.sum

    float correct / float x.RowCount

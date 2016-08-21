module ML.GD.NNGradient

open MathNet.Numerics.LinearAlgebra
open ML.Core.LinearAlgebra

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

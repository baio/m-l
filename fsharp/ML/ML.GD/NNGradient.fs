module ML.GD.NNGradient

open MathNet.Numerics.LinearAlgebra
open ML.Core.LinearAlgebra

open GLM
open ML.NN

let NNCost (thetaShape: ThetaShape) (x : FMatrix) (y : FVector) (theta: FVector)  =     
    let shape = thetaShape.nnShape()
    let errSum = 
        x.EnumerateRows() 
        |> Seq.map(fun row ->
            let out = forward row shape theta
            (out - y).PointwisePower(2.) |> Vector.sum
        )
        |> Seq.sum
    errSum / (2. * float x.RowCount) 
        
let NNGradient (thetaShape: ThetaShape) (x : FMatrix) (y : FVector) (theta: FVector) =
    let shape = thetaShape.nnShape()
    let gradSum = 
        x.EnumerateRows()
        |> Seq.map(fun f ->
            backprop y f shape theta
            |> flatNNGradients
        )
        |> DenseMatrix.ofColumnSeq
        |> Matrix.sumCols
    gradSum / float x.RowCount

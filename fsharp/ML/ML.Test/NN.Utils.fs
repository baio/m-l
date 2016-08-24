module ML.NN.Utils.Test

open Xunit
open FsUnit
open MathNet.Numerics.LinearAlgebra

open ML.NN
open ML.Core.LinearAlgebra

let f a = a
let act = {f = f; f' = f}


[<Fact>]
let ``Chunk outputs must work``() =
    
    let mx = matrix [[0.; 1.]; [1.; 0.]; [7.; 3.;]] 
    let actual = mx |> flatMx |> chunkOutputs mx.RowCount
    
    actual |> should equal mx

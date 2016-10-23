module ML.NN.Forward.Test

open Xunit
open FsUnit
open ML.NN
open MathNet.Numerics.LinearAlgebra

open ML.Core.LinearAlgebra

let f a = a
let act = {f = f; f' = f}

[<Fact>]
let ``embed@2,1 [1 1 1 1] -> (1; 2; 3; 4) -> [3, 7]``() =

    let shape = {
        Layers = [
            NNFullLayerShape({NodesNumber = 4; Activation = act});
            NNEmbedLayerShape({BlocksNumber = 2; NodesInBlockNumber = 1; Activation = act});
        ]
    }

    let inputs = matrix [[1.;1.;1.;1.]] 
    let thetas = vector [1.;2.;3.;4.]

    let actual = forwardOutput inputs shape thetas
    let expected = matrix [[3.; 7.]]

    actual |> should equal expected



[<Fact>]
let ``embed@2,2 [1 1 2 2] -> (1; 2; 3; 4) -> [[3, 7], [6, 14]]``() =

    let shape = {
        Layers = [
            NNFullLayerShape({NodesNumber = 4; Activation = act});
            NNEmbedLayerShape({BlocksNumber = 2; NodesInBlockNumber = 2; Activation = act});
        ]
    }

    let inputs = [[1.;1.;2.;2.]] |> DenseMatrix.ofRowList
    let thetas = vector [1.;3.;2.;4.;1.;3.;2.;4.]

    System.Diagnostics.Debug.WriteLine(sprintf "%A" inputs)

    let actual = forwardOutput inputs shape thetas
    let expected = matrix [[3.; 7.;6.; 14.]]

    actual |> should equal expected


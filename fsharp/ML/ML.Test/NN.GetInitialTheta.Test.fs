module ML.NN.GetInitialTheta.Test

open Xunit
open FsUnit
open ML.NN
open MathNet.Numerics.LinearAlgebra

open ML.Core.LinearAlgebra

let f a = a
let act = {f = f; f' = f}

[<Fact>]
let ``gen initial theta for embed [4] -> [1, 1]``() =

    let shape = {
        Layers = [
            NNFullLayerShape({NodesNumber = 4; Activation = act});
            NNEmbedLayerShape({BlocksNumber = 2; NodesInBlockNumber = 1; Activation = act});
        ]
    }

    let thetas = getInitialTheta shape

    thetas.[0] |> should equal thetas.[2]
    thetas.[1] |> should equal thetas.[3]



[<Fact>]
let ``gen initial theta for embed [4] -> [2, 2]``() =

    let shape = {
        Layers = [
            NNFullLayerShape({NodesNumber = 4; Activation = act});
            NNEmbedLayerShape({BlocksNumber = 2; NodesInBlockNumber = 2; Activation = act});
        ]
    }

    let thetas = getInitialTheta shape

    thetas.[0] |> should equal thetas.[4]
    thetas.[1] |> should equal thetas.[5]
    thetas.[2] |> should equal thetas.[6]
    thetas.[3] |> should equal thetas.[7]

module ML.NN.Forward.Embed.Test

open NUnit.Framework
open FsUnit
open ML.NN
open MathNet.Numerics.LinearAlgebra

open ML.Core.LinearAlgebra

let f a = a
let act = {f = f; f' = f}

// TODO : we should calcualte gardients universaly
// We should initializae / update weights distinctly for each type of layer

//[<TestCase>]
let ``forward : [1; 1; 1; 1;] -> <embed(2,1)>([1; 2;] [3; 4]) -> {3, 7}``() =

    let shape = {
        Layers = 
        [
            NNFullLayerShape({NodesNumber = 4; Activation = act});
            NNEmbedLayerShape({BlocksNumber = 2; NodesInBlockNumber = 1; Activation = act});
        ]
    }

    let inputs = matrix [[1.;1.;1.;1.]] 
    let thetas = vector [1.;2.;3.;4.]

    let actual = forwardOutput inputs shape thetas
    let expected = matrix [[3.; 7.]]

    actual |> should equal expected

//[<TestCase>]
let ``forawrd : [1; 1; 2; 2;] -> <embed(2,2)>([1; 2; 3; 4;] [1; 2; 3; 4;]) -> {[3, 7], [6, 14]}``() =

    let shape = {
        Layers = 
        [
            NNFullLayerShape({NodesNumber = 4; Activation = act});
            NNEmbedLayerShape({BlocksNumber = 2; NodesInBlockNumber = 2; Activation = act});
        ]
    }

    let inputs = [[1.;1.;2.;2.]] |> DenseMatrix.ofRowList
    let thetas = vector [1.;3.;2.;4.;1.;3.;2.;4.]

    let actual = forwardOutput inputs shape thetas
    let expected = matrix [[3.; 7.;6.; 14.]]

    actual |> should equal expected


module ML.NN.Backprop.Embed.Test

open NUnit.Framework
open FsUnit
open ML.NN
open ML.GradientCheck
open MathNet.Numerics.LinearAlgebra

open ML.Core.Utils
open ML.Core.LinearAlgebra

open NN.Tests.NUnit.Utils

let f a = a
let f' a = a |> Vector.length |> ones
let act = {f = f; f' = f'}

//[<TestCase>]
let ``backprop : [1;2;3;4] -> <embed(2,1)>([1;1] [1;1]) -> [h1; h2] -> (0;1;1) -> {5}``() =

    let shape = {
        Layers = 
        [
            NNFullLayerShape({NodesNumber = 4; Activation = act});
            NNEmbedLayerShape({BlocksNumber = 2; NodesInBlockNumber = 1; Activation = act});
            NNFullLayerShape({NodesNumber = 1; Activation = act});
        ]
    }

    let y = vector [5.]
    let x = matrix [[1.; 2.; 3.; 4.;]]
    let theta = vector([1.;1.;1.;1.;0.;1.;1.])
    
    let bkprp = backprop y x shape theta |> Seq.toArray
    let grad = gradCheck y (x.Row 0) shape theta 1E-4 |> Seq.toArray

    dprintf bkprp
    dprintf grad

    bkprp |> should equal (grad +/- 0.5)



//[<TestCase>]
let ``calc backprop and grad check``() =

    let shape = {
        Layers = 
        [
            NNFullLayerShape({NodesNumber = 4; Activation = act});
            NNEmbedLayerShape({BlocksNumber = 2; NodesInBlockNumber = 1; Activation = act});
            NNFullLayerShape({NodesNumber = 1; Activation = act});
        ]
    }

    let y = vector [10.]

    let x = matrix [[1.; 1.; 2.; 2.;]]
    let theta = vector([0.;1.;2.;3.;0.;1.;2.])

    let grad = gradCheck y (x.Row 0) shape theta 1E-4
    
    let actual = backprop y x shape theta
    let expected = vector [1.5; 1.5; 27.5; 27.5; 11.; 11.; 110.]

    actual |> should equal expected










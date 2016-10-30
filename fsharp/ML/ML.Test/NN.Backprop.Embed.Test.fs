module ML.NN.Backprop.Embed.Test

open Xunit
open FsUnit
open ML.NN
open ML.GradientCheck
open MathNet.Numerics.LinearAlgebra


open ML.Core.LinearAlgebra

let f a = a
let f' a = a |> Vector.length |> ones
let act = {f = f; f' = f'}

[<Fact>]
let ``backprop : [1;1;2;2] -> <embed(2,1)>([1;1] [1;1]) -> [h1; h2] -> (0;1;2) -> {y}``() =

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
    let theta = vector([1.;1.;1.;1.;0.;1.;2.])
    
    let bkprp = backprop y x shape theta
    let grad = gradCheck y (x.Row 0) shape theta 1E-4

    bkprp |> should equal grad


//[<Fact>]
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









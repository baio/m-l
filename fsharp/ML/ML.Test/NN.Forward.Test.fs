module ML.NN.Forward.Test

open Xunit
open FsUnit
open ML.NN
open ML.NN.Backprop
open MathNet.Numerics.LinearAlgebra
open ML.Core.LinearAlgebra
open ML.Core.Utils

let f a = a
let f' (a: FVector) = ones a.Count
let act = {f = f; f' = f'}


[<Fact>]
let ``forward: [1; 1;] -> (0; 1; 2; 0; 1; 2) -> [1; 1;] -> (0, 1, 2) -> {x}``() =

    let x = matrix([[1.; 1.]])
    let theta = vector([0.; 1.; 2.; 0.; 1.; 2.; 0.; 1.; 2.])
    
    let shape = {
        Layers = 
            [ 
                NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
                NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
                NNFullLayerShape({ NodesNumber = 1; Activation = act }); 
            ]
    }
    
    let fwd = forward x shape theta |> Seq.toArray

    dprintf fwd
    
    fwd |> should equal fwd

module ML.NN.Backprop.Test

open Xunit
open FsUnit
open MathNet.Numerics.LinearAlgebra

open ML.NN
open ML.GradientCheck

open ML.GD.LogisticRegression

open ML.Core.Readers
open ML.Core.Utils
open ML.Core.LinearAlgebra
open ML.GD.GLM
open ML.GD.NNGradient
open ML.GD.GradientDescent


open ML.GD.GD
open ML.GD.SGD

let f a = a
let f' (a: FVector) = ones a.Count
let act = {f = f; f' = f'}
let sigm = { f= sigmoid;  f' = sigmoid'} 


//[<Fact>]
let ``backprop + gradcheck => [1;1] -> (1;1) -> 5``() =
    
    let x = matrix([[1.; 1.]])
    let y = vector([5.])
    let theta = vector([0.; 1.; 1.])
    
    let shape = {
        Layers = 
            [ 
                NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
                NNFullLayerShape({ NodesNumber = 1; Activation = act }); 
            ]
    }
    
    let bkprp = backprop y x shape theta
    let chk = gradCheck y (x.Row 0) shape theta 1E-4

    dprintf bkprp
    dprintf chk

    bkprp |> should equal chk

[<Fact>]
let ``backprop + gradcheck => [1;1] -> (1;1) -> (1;1) -> 5``() =
    
    let x = matrix([[1.; 1.]])
    let y = vector([5.])
    let theta = vector([0.; 0.; 1.; 1.; 1.; 1.; 0.; 0.; 1.; 1.; 1.; 1.; 0.; 1.; 1.])
    
    let shape = {
        Layers = 
            [ 
                NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
                NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
                NNFullLayerShape({ NodesNumber = 1; Activation = act }); 
            ]
    }
    
    let bkprp = backprop y x shape theta |> Seq.toArray
    let chk = gradCheck y (x.Row 0) shape theta 1E-4 |> Seq.toArray

    dprintf bkprp
    dprintf chk

    bkprp |> should equal chk

//[<Fact>]
let ``XOR Backprop check witg GradCheck``() =
    
    // XOR, Math from Artificial Network 
    let x = matrix([[0.05; 0.10]])
    let y = vector([0.01; 0.99;])
    let theta = vector([0.35; 0.35; 0.15; 0.25; 0.20; 0.30;  0.6; 0.6; 0.4; 0.5; 0.45; 0.55])
    
    let shape = {
        Layers = 
            [ 
                NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
                NNFullLayerShape({ NodesNumber = 2; Activation = sigm }); 
                NNFullLayerShape({ NodesNumber = 2; Activation = sigm }); 
            ]
    }
    
    let bkprp = backprop y x shape theta
    let grdCheck = gradCheck y (x.Row 0) shape theta 1E-4

    dprintf bkprp
    dprintf grdCheck

    bkprp |> should equal grdCheck

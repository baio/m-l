module ML.NN.Backprop.Test

open NUnit.Framework
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

open NN.Tests.NUnit.Utils

let f a = a
let f' (a: FVector) = ones a.Count
let act = {f = f; f' = f'}
let sigm = { f= sigmoid;  f' = sigmoid'} 

// TODO : Approx equality
// TODO : Get rid of input layer in NNShape 
// TODO : Define use bias explicitly for each layer

[<TestCase>]
let ``backprop : [1;1] -> (0;1;1) -> {5}``() =
    
    let x = matrix([[1.; 1.]])
    let y = vector([5.])
    let theta = vector([0.; 1.; 1.;])
    
    let shape = {
        Layers = 
            [ 
                //First (input layer), 2 nodes. TODO : Create NNInputLayerShape
                NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
                //Second (output layer) fully connected, 1 node
                NNFullLayerShape({ NodesNumber = 1; Activation = act }); 
            ]
    }
    
    let bkprp = backprop y x shape theta |> Seq.toArray
    let chk = gradCheck y (x.Row 0) shape theta 1E-4 |> Seq.toArray

    dprintf bkprp
    dprintf chk

    bkprp |> should equal (chk +/- 1E-10)

[<TestCase>]
let ``backprop + gradcheck => [1;1] -> (0;1;1;0;1;1) -> [1;1] -> (0;1;1)  -> {5}``() =
    
    let x = matrix([[1.; 1.]])
    let y = vector([5.])
    let theta = vector([0.;1.;1.;0.;1.;1.;0.;1.;1.])
    
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

    bkprp |> should equal (chk +/- 1E-10)

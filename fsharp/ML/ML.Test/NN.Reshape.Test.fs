﻿module ML.NN.Reshape.Test

open NUnit.Framework
open FsUnit
open MathNet.Numerics.LinearAlgebra

open ML.NN
open ML.Core.Utils

let f a = a
let act = {f = f; f' = f}


[<TestCase>]
let ``Reshape for 2 inputs -> 1 output must work``() =
    
    let theta = vector([0.; 1.; 2.])
    
    let shape = {
        Layers = 
            [ 
                NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
                NNFullLayerShape({ NodesNumber = 1; Activation = act }); 
            ]
    }
    
    let actual = reshapeNN shape theta
    let expected = matrix([ [0.; 1.; 2.] ]);
    
    actual.[0].Thetas |> should equal expected

[<TestCase>]
let ``Reshape for 3 inputs -> 2 hidden -> 1 output must work``() =
    
    let theta = vector([0.; 1.; 2.; 3.;  4.; 5.; 6.; 7.;  8.; 9.; 10.])
    
    let shape = {
        Layers = 
        [ 
            NNFullLayerShape({ NodesNumber = 3; Activation = act }); 
            NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
            NNFullLayerShape({ NodesNumber = 1; Activation = act }); 
        ]
    }
    
    let actual = reshapeNN shape theta
    let expected = 
        [| 
            matrix([ [0.; 2.; 4.; 6.]; [1.; 3.; 5.; 7.]])
            matrix([ [8.; 9.; 10.] ])
        |]
    
    actual.[0].Thetas |> should equal expected.[0]
    actual.[1].Thetas |> should equal expected.[1]

[<TestCase>]
let ``Reshape XOR must work``() =
    
    let theta = vector([-10.; -30.; 20.; 20.; 20.; 20.; 10.; 20.; -20.])
    
    let shape = {
        Layers = 
        [ 
            NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
            NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
            NNFullLayerShape({ NodesNumber = 1; Activation = act }); 
        ]
    }
    
    let actual = reshapeNN shape theta
    let expected = 
        [| 
            matrix([ [-10.; 20.; 20.;]; [-30.; 20.; 20.;] ])
            matrix([ [10.; 20.; -20.] ])
        |]
    
    actual.[0].Thetas |> should equal expected.[0]
    actual.[1].Thetas |> should equal expected.[1]




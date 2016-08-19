module ML.NN.Clac.Test

open Xunit
open FsUnit
open MathNet.Numerics.LinearAlgebra

open ML.NN.Types
open ML.NN.NN

open ML.Regressions.LogisticRegression

let f a = a
let act = {f = f; f' = f}
let sigm = { f= sigmoid;  f' = sigmoid'} 

[<Fact>]
let ``Calc 1 input -> 1 output must work``() =
    
    let inputs = vector([2.])
    let theta = vector([10.; 20.])
    
    let shape = {
        Layers = [ 
            { NodesNumber = 1; Activation = act }; 
            { NodesNumber = 1; Activation = act }; 
        ]
    }
    
    let actual = forward inputs shape theta
    let expected = vector([50.]);
    
    actual |> should equal expected

[<Fact>]
let ``Calc XOR Layer1 must work``() =
    
    // XOR, Math from Artificial Network 
    let inputs = vector([1.; 0.])
    let theta = vector([-0.46; 0.10; -0.07; 0.94; 0.22; 0.46;])
    
    let shape = {
        Layers = [ 
            { NodesNumber = 2; Activation = act }; 
            { NodesNumber = 2; Activation = sigm }; 
        ]
    }
    
    let actual = forward inputs shape theta
    let expected = vector([0.37; 0.74]);
    
    (actual |> Vector.map (fun m -> System.Math.Round(m, 2))) 
    |> should equal expected

[<Fact>]
let ``Calc XOR must work``() =
    
    // XOR, Math from Artificial Network 
    let inputs = vector([1.; 0.])
    let theta = vector([-0.46; 0.10; -0.07; 0.94; 0.22; 0.46; 0.78; -0.22; 0.58])
    
    let shape = {
        Layers = [ 
            { NodesNumber = 2; Activation = act }; 
            { NodesNumber = 2; Activation = sigm }; 
            { NodesNumber = 1; Activation = sigm }; 
        ]
    }
    
    let actual = forward inputs shape theta
    let expected = vector([0.76]);
    
    (actual |> Vector.map (fun m -> System.Math.Round(m, 2)))
    |> should equal expected


[<Fact>]
let ``Calc Example must work``() =
    
    // XOR, Math from Artificial Network 
    let inputs = vector([0.05; 0.10])
    let theta = vector([0.35; 0.35; 0.15; 0.25; 0.20; 0.30;  0.6; 0.6; 0.4; 0.5; 0.45; 0.55])
    
    let shape = {
        Layers = [ 
            { NodesNumber = 2; Activation = act }; 
            { NodesNumber = 2; Activation = sigm }; 
            { NodesNumber = 2; Activation = sigm }; 
        ]
    }
    
    let actual = forward inputs shape theta
    let expected = vector([0.75; 0.77]);
    
    (actual |> Vector.map (fun m -> System.Math.Round(m, 2)))
    |> should equal expected


[<Fact>]
let ``Calc Example grads must work``() =
    
    // XOR, Math from Artificial Network 
    let inputs = vector([0.05; 0.10])
    let outputs = vector([0.01; 0.99])
    let theta = vector([0.35; 0.35; 0.15; 0.25; 0.20; 0.30;  0.6; 0.6; 0.4; 0.5; 0.45; 0.55])
    
    let shape = {
        Layers = [ 
            { NodesNumber = 2; Activation = act }; 
            { NodesNumber = 2; Activation = sigm }; 
            { NodesNumber = 2; Activation = sigm }; 
        ]
    }
    
    let actual = backprop outputs inputs shape theta
    let expected = vector([0.75; 0.77]);
        
    true |> should equal false

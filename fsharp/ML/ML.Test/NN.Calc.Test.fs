module ML.NN.Clac.Test

open NUnit.Framework
open FsUnit
open MathNet.Numerics.LinearAlgebra

open ML.NN

open ML.GD.LogisticRegression

open ML.Core.Readers
open ML.Core.Utils
open ML.GD.GLM
open ML.GD.NNGradient
open ML.GD.GradientDescent


open ML.GD.GD
open ML.GD.SGD

let f a = a
let act = {f = f; f' = f}
let sigm = { f= sigmoid;  f' = sigmoid'} 

[<TestCase>]
let ``Calc 1 input -> 1 output must work``() =
    
    let inputs = matrix([[2.]])
    let theta = vector([10.; 20.])
    
    let shape = {
        Layers = 
            [ 
                NNFullLayerShape({ NodesNumber = 1; Activation = act }); 
                NNFullLayerShape({ NodesNumber = 1; Activation = act }); 
            ]
    }
    
    let actual = forwardOutput inputs shape theta
    let expected = matrix([[50.]]);
    
    actual |> should equal expected

[<TestCase>]
let ``Calc XOR Layer1 must work``() =
    
    // XOR, Math from Artificial Network 
    let inputs = matrix([[1.; 0.]])
    let theta = vector([-0.46; 0.10; -0.07; 0.94; 0.22; 0.46;])
    
    let shape = {
        Layers = 
            [ 
                NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
                NNFullLayerShape({ NodesNumber = 2; Activation = sigm }); 
            ]
    }
    
    let actual = forwardOutput inputs shape theta
    let expected = matrix([[0.37; 0.74]]);
    
    (actual |> Matrix.map (fun m -> System.Math.Round(m, 2))) 
    |> should equal expected

[<TestCase>]
let ``Calc XOR must work``() =
    
    // XOR, Math from Artificial Network 
    let inputs = matrix([[1.; 0.]])
    let theta = vector([-0.46; 0.10; -0.07; 0.94; 0.22; 0.46; 0.78; -0.22; 0.58])
    
    let shape = {
        Layers = 
            [ 
                NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
                NNFullLayerShape({ NodesNumber = 2; Activation = sigm }); 
                NNFullLayerShape({ NodesNumber = 1; Activation = sigm }); 
            ]
    }
    
    let actual = forwardOutput inputs shape theta
    let expected = matrix([[0.76]]);
    
    (actual |> Matrix.map (fun m -> System.Math.Round(m, 2)))
    |> should equal expected


[<TestCase>]
let ``Calc Example must work``() =
    
    // XOR, Math from Artificial Network 
    let inputs = matrix([[0.05; 0.10]])
    let theta = vector([0.35; 0.35; 0.15; 0.25; 0.20; 0.30;  0.6; 0.6; 0.4; 0.5; 0.45; 0.55])
    
    let shape = {
        Layers = 
            [ 
                NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
                NNFullLayerShape({ NodesNumber = 2; Activation = sigm }); 
                NNFullLayerShape({ NodesNumber = 2; Activation = sigm }); 
            ]
    }
    
    let actual = forwardOutput inputs shape theta
    let expected = matrix([[0.75; 0.77]]);
    
    (actual |> Matrix.map (fun m -> System.Math.Round(m, 2)))
    |> should equal expected


[<TestCase>]
let ``Calc Example grads must work``() =
    
    // XOR, Math from Artificial Network 
    let inputs = matrix([[0.05; 0.10]])
    let outputs = vector([0.01; 0.99])
    let theta = vector([0.35; 0.35; 0.15; 0.25; 0.20; 0.30;  0.6; 0.6; 0.4; 0.5; 0.45; 0.55])
    
    let shape = {
        Layers = 
            [ 
                NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
                NNFullLayerShape({ NodesNumber = 2; Activation = sigm }); 
                NNFullLayerShape({ NodesNumber = 2; Activation = sigm }); 
            ]
    }
    
    let actual = backprop outputs inputs shape theta

    let expected = 
        [

            0.0087713546894869366
            0.0099542547052172015
            0.00043856773447434685
            0.0004977127352608601
            0.0008771354689486937
            0.0009954254705217202

            0.13849856162855698
            -0.038098236516556229
            0.082167040564230784
            -0.022602540477475067
            0.082667627847533259
            -0.022740242215978219
        ] 

    actual |> should equal expected

[<TestCase>]
let ``Calc GD for Example NN must work``() =
    
    //TODO : Wee need output as Vector NOT as float !
    //There is no way set NN output as a Vector
    let inputs = matrix [[0.05; 0.10]]
    let outputs = vector [0.01; 0.99]
    
    let theta = vector([0.35; 0.35; 0.15; 0.25; 0.20; 0.30;  0.6; 0.6; 0.4; 0.5; 0.45; 0.55])
    
    let model = {
        Cost = NNCost
        Gradient = NNGradient
    }

    let prms = {
        EpochNumber = 1
        ConvergeMode = ConvergeModeNone
    }       

    let stochasticHyper : SGDHyperParams = {
        Alpha = 0.5
        BatchSize = 1
    }

    let shape = 
        {
            Layers = 
                [ 
                    NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
                    NNFullLayerShape({ NodesNumber = 2; Activation = sigm }); 
                    NNFullLayerShape({ NodesNumber = 2; Activation = sigm }); 
                ]
        }

    let glmModel = 
        GLMNNModel(
            { 
                Base = model; 
                Shape = shape; 
                InitialTheta = Some(theta) 
            }
        )

    let expected = [
        [[
            [0.345614;  0.149781;  0.199561]
            [0.345023;  0.249751;  0.299502]
        ] |> DenseMatrix.ofRowList]
        [[
            [0.530751;  0.358916;  0.408666]
            [0.619049;  0.511301;   0.56137]
        ] |> DenseMatrix.ofRowList]
    ]

    let gd = stochasticHyper |> SGDHyperParams |> gradientDescent glmModel prms inputs outputs 

    let th = gd.Theta |> Vector.map (fun f -> System.Math.Round(f, 6))
    
    let actual = reshapeNN shape th |> Array.map (fun f -> f.Thetas) |> Array.toList

    System.Diagnostics.Debug.WriteLine(sprintf "%A" expected)
    System.Diagnostics.Debug.WriteLine(sprintf "%A" actual)
    
    actual |> should equal expected


[<TestCase>]
let ``Calc Example grads must work with 2 samples``() =
    
    // XOR, Math from Artificial Network 
    let inputs = matrix([[0.05; 0.10]; [0.05; 0.10]])
    let outputs = vector([0.01; 0.01; 0.99; 0.99])
    let theta = vector([0.35; 0.35; 0.15; 0.25; 0.20; 0.30;  0.6; 0.6; 0.4; 0.5; 0.45; 0.55])
    
    let shape = {
        Layers = 
            [ 
                NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
                NNFullLayerShape({ NodesNumber = 2; Activation = sigm }); 
                NNFullLayerShape({ NodesNumber = 2; Activation = sigm }); 
            ]
    }
    
    let actual = backprop outputs inputs shape theta

    let expected = 
        [

            0.0087713546894869366
            0.0099542547052172015
            0.00043856773447434685
            0.0004977127352608601
            0.0008771354689486937
            0.0009954254705217202

            0.13849856162855698
            -0.038098236516556229
            0.082167040564230784
            -0.022602540477475067
            0.082667627847533259
            -0.022740242215978219
        ] 

    actual |> should equal expected

//[<TestCase>]
let ``XOR must work``() =

    let rnd = new System.Random()
    let generator = List.init 100 (fun _ ->
        let x1, x2 = (rnd.Next(2) |> float), (rnd.Next(2) |> float)
        let y = 
            match x1, x2 with
            | (1., 1.) | (0., 0.) -> 0.
            | _ -> 1.
        [y; x1; x2;]
    )

    let mx = 
        generator |> DenseMatrix.ofRowList

    let y = mx.Column(0)
    let x = mx.RemoveColumn(0)

    let model = {
        Cost = NNCost
        Gradient = NNGradient
    }

    let prms = {
        EpochNumber = 10
        ConvergeMode = ConvergeModeNone
    }       

    let stochasticHyper : SGDHyperParams = {
        Alpha = 0.5
        BatchSize = 1
    }

    let shape = 
        {
            Layers = 
                [ 
                    NNFullLayerShape({ NodesNumber = 2; Activation = act }); 
                    NNFullLayerShape({ NodesNumber = 2; Activation = sigm }); 
                    NNFullLayerShape({ NodesNumber = 1; Activation = sigm }); 
                ]
        }

    let glmModel = 
        GLMNNModel(
            { 
                Base = model; 
                Shape = shape; 
                InitialTheta = None
            }
        )
    
    let gd = stochasticHyper |> SGDHyperParams |> gradientDescent glmModel prms x y

    true |> should equal true

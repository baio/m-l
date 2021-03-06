﻿module NN_XOR

open MathNet.Numerics.LinearAlgebra

open ML.NN

open ML.GD.LogisticRegression

open ML.Core.Readers
open ML.Core.Utils
open ML.Core.LinearAlgebra
open ML.GD.GLM
open ML.GD.NNGradient
open ML.GD.GradientDescent


open ML.GD.GD
open ML.GD.SGD
open ML.GD.NAG
open ML.GD.Adagrad
open ML.GD.Adadelta


let f a = a
let act = {f = f; f' = f}
let sigm = { f= sigmoid;  f' = sigmoid'} 

open PerfUtil

open ML.Statistics.Regressions
open ML.Statistics.Charting

let nn_xor() = 
        
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
        EpochNumber = 100
        ConvergeMode = ConvergeModeNone
    }       


    let shape = 
        {
            Layers = 
                [ 
                    { NodesNumber = 2; Activation = act }; 
                    { NodesNumber = 2; Activation = sigm }; 
                    { NodesNumber = 1; Activation = sigm }; 
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

    ///
    let stochasticHyper : SGDHyperParams = {
        Alpha = 0.5
        BatchSize = 1
    }

    let batchHyper : SGDHyperParams = {
        Alpha = 0.5
        BatchSize = x.RowCount
    }

    let minbatchHyper : SGDHyperParams = {
        Alpha = 0.5
        BatchSize = 5
    }

    let NAGHyper : NAGHyperParams = {        
        Alpha = 0.01
        BatchSize = 5
        Gamma = 0.5
    }

    let AdagradHyper : AdagradHyperParams = {        
        Alpha = 0.01
        BatchSize = 5
        Epsilon = 1E-8
    }

    let AdadeltaHyper : AdadeltaHyperParams = {        
        BatchSize = 5
        Epsilon = 1E-8
        Rho = 0.6
    }

    ///

    let gd = gradientDescent glmModel prms x y

    let mutable trainResults = [] 

    let perf = Benchmark.Run (fun () ->
        let train = batchHyper |> SGDHyperParams |> gd
        trainResults <- ("batch",train)::trainResults
        printfn "batch result : %A" train
    )
    printfn "batch perf : %A" perf

    (*
    let perf = Benchmark.Run (fun () ->
        let train = stochasticHyper |> SGDHyperParams |> gd
        trainResults <- ("stochastic", train)::trainResults
        printfn "stochastic result : %A" train
    )    
    printfn "stochastic perf : %A" perf
    *)
    
    let perf = Benchmark.Run (fun () ->
        let train = minbatchHyper |> SGDHyperParams |> gd
        trainResults <- ("miniBatch", train)::trainResults
        printfn "miniBatch result : %A" train
    )    
    printfn "miniBatch perf : %A" perf

    let perf = Benchmark.Run (fun () ->
        let train = NAGHyper |> NAGHyperParams |> gd
        trainResults <- ("NAG", train)::trainResults
        printfn "NAG result : %A" train
    )    
    printfn "NAG perf : %A" perf
    

    let perf = Benchmark.Run (fun () ->
        let train = AdagradHyper |> AdagradHyperParams |> gd
        trainResults <- ("Adagrad", train)::trainResults
        printfn "Adagrad result : %A" train
    )    
    printfn "Adagrad perf : %A" perf

    let perf = Benchmark.Run (fun () ->
        let train = AdadeltaHyper |> AdadeltaHyperParams |> gd
        trainResults <- ("Adadelta", train)::trainResults
        printfn "Adadelta result : %A" train
    )    
    printfn "Adadelta perf : %A" perf

    let mapOutput = (fun (f: FVector) -> [iif (f.At(0) < 0.5) 0. 1.] |> vector)
    let acc = accuracy mapOutput shape x y 

    trainResults
    |> List.sortBy (fun (_, res) -> res.Errors.[0])
    |> List.map (fun (label, res) -> (sprintf "%s %f : %f (%i)" label (acc res.Theta) res.Errors.[0] res.Errors.Length), res.Errors |> List.mapi(fun i x -> (float i, x)))        
    |> showLines2
    
﻿module Softmax

open ML.Core.Readers
open ML.Core.Utils

open ML.Regressions.GLM
open ML.Regressions.SoftmaxRegression
open ML.Regressions.GD
open ML.Regressions.SGD
open ML.Regressions.NAG
open ML.Regressions.Adagrad
open ML.Regressions.Adadelta

open ML.Regressions.GradientDescent

open MathNet.Numerics.LinearAlgebra
open PerfUtil

open ML.Statistics.Regressions
open ML.Statistics.Charting

let softmax() = 
        
    let inputs, outputs = readCSV @"..\..\..\..\..\machine-learning-ex1\ex1\ex1data2.csv" false [|0..1|] 2    
    let outputs = vector outputs
    let inputs = matrix inputs
    let inputs, normPrms = norm inputs

    let model : GLMSoftmaxModel = {
        Base = { Cost = softmaxCost; Gradient = softmaxGradient }
        ClassesNumber = 10
    }

    let prms = {
        EpochNumber = 5000 // Epochs number
        ConvergeMode = ConvergeModeCostStopsChange
    }

    let batchHyper : SGDHyperParams = {
        Alpha = 0.01
        BatchSize = inputs.RowCount
    }

    let stochasticHyper : SGDHyperParams = {
        Alpha = 0.01
        BatchSize = 1
    }

    let SGDHyper : SGDHyperParams = {
        Alpha = 0.01
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

    let mutable trainResults = [] 

    let gd = gradientDescent (GLMSoftmaxModel model) prms inputs outputs 
    
    let perf = Benchmark.Run (fun () ->
        let train = SGDHyperParams batchHyper |> gd
        trainResults <- ("batch",train)::trainResults
        printfn "batch result : %A" train
    )    
    printfn "batch perf : %A" perf
    
    let perf = Benchmark.Run (fun () ->
        let train = SGDHyperParams stochasticHyper |> gd
        trainResults <- ("stochastic", train)::trainResults
        printfn "stochastic result : %A" train
    )    
    printfn "stochastic perf : %A" perf

    let perf = Benchmark.Run (fun () ->
        let train = SGDHyperParams SGDHyper |> gd
        trainResults <- ("SGD", train)::trainResults
        printfn "miniBatch result : %A" train
    )    
    printfn "miniBatch perf : %A" perf

    let perf = Benchmark.Run (fun () ->
        let train = NAGHyperParams NAGHyper |> gd
        trainResults <- ("NAG", train)::trainResults
        printfn "NAG result : %A" train
    )    
    printfn "NAG perf : %A" perf

    
    let perf = Benchmark.Run (fun () ->
        let train = AdagradHyperParams AdagradHyper |> gd
        trainResults <- ("Adagrad", train)::trainResults
        printfn "Adagrad result : %A" train
    )    
    printfn "Adagrad perf : %A" perf
    
    let perf = Benchmark.Run (fun () ->
        let train = AdadeltaHyperParams AdadeltaHyper |> gd
        trainResults <- ("Adadelta", train)::trainResults
        printfn "Adadelta result : %A" train
    )    
    printfn "Adadelta perf : %A" perf
        
    trainResults
    |> List.sortBy (fun (_, res) -> res.Errors.[0])
    |> List.map (fun (label, res) -> (sprintf "%s : %f (%i)" label res.Errors.[0] res.Errors.Length), res.Errors |> List.mapi(fun i x -> (float i, x)))        
    |> showLines2

    

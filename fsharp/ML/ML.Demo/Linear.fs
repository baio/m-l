﻿// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
module Linear

open ML.Core.Readers
open ML.Core.Utils
open ML.Regressions.GLM
open ML.Regressions.LinearRegression
open ML.Regressions.BatchGradientDescent
open ML.Regressions.StochasticGradientDescent
open ML.Regressions.MiniBatchGradientDescent
open ML.Regressions.NesterovAcceleratedGradient

open MathNet.Numerics.LinearAlgebra
open PerfUtil

open ML.Statistics.Regressions

let linear() = 
        
    let _inputs, outputs = readCSV @"..\..\..\..\..\machine-learning-ex1\ex1\ex1data2.csv" false [|0..1|] 2    
    let outputs = vector outputs
    let inputs = matrix _inputs
    let inputs, normPrms = norm inputs
    let model = {
        Hypothesis = linearHyp
        Loss = linearMSELoss
        Gradient = linearMSEGradient
    }
    let prms = {
        MaxIterNumber = 5000 // Epochs number
        MinErrorThreshold = 0.
        Alpha = 0.01 
    }
    let minBatchPrms : MiniBatchTrainModelParams = {
        MaxIterNumber = 5000 // Epochs number
        MinErrorThreshold = 0.
        Alpha = 0.01
        BatchSize = 5
    }
    let acceleratedBatchPrms : AcceleratedTrainModelParams = {
        EpochNumber = 50000 // Epochs number
        MinErrorThreshold = 0.
        Alpha = 0.01
        BatchSize = 5
        Gamma = 0.009
    }

    let perf = Benchmark.Run (fun () ->
        let train = batchGradientDescent model prms inputs outputs        
        printfn "batch result : %A" train
    )    
    printfn "batch perf : %A" perf

    let perf = Benchmark.Run (fun () ->
        let train = stochasticGradientDescent model prms inputs outputs
        printfn "stochastic result : %A" train
    )    
    printfn "stochastic perf : %A" perf

    let perf = Benchmark.Run (fun () ->
        let train = miniBatchGradientDescent model minBatchPrms inputs outputs
        printfn "miniBatch result : %A" train
    )    
    printfn "miniBatch perf : %A" perf

    let perf = Benchmark.Run (fun () ->
        let train = nesterovAcceleratedGradientDescent model acceleratedBatchPrms inputs outputs
        printfn "nesterove result : %A" train
    )    
    printfn "nesterov perf : %A" perf

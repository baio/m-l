// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
module Logistic

open ML.Core.Readers
open ML.Core.Utils
open ML.Regressions.GLM
open ML.Regressions.LogisticRegression
open ML.Regressions.BatchGradientDescent
open ML.Regressions.StochasticGradientDescent
open ML.Regressions.MiniBatchGradientDescent
open ML.Regressions.NesterovAcceleratedGradient

open MathNet.Numerics.LinearAlgebra
open PerfUtil

open ML.Statistics.Regressions

open ML.Core.LinearAlgebra

let logistic() = 
            
    let _inputs, outputs = readCSV @"..\..\..\..\..\machine-learning-ex2\ex2\ex2data1.txt" false [|0..1|] 2    
    let outputs = vector outputs
    let inputs = matrix _inputs
    let inputs, normPrms = norm inputs
    let model = {
        Hypothesis = logisticHyp
        Loss = logisticMSELoss
        Gradient = logisticMSEGradient
    }
    let prms = {
        MaxIterNumber = 400 // Epochs number
        MinErrorThreshold = 0.
        Alpha = 0.01
    }       
    
    let x = inputs |> appendOnes
    let w = zeros x.ColumnCount
    let loss0 = logisticMSELoss w x outputs
    let theta0 = logisticMSEGradient w x outputs
    printfn "=== %A \n %A" loss0 theta0

    let perf = Benchmark.Run (fun () ->
        let train = batchGradientDescent model prms inputs outputs        
        printfn "batch result : %A" train
    )    
    printfn "batch perf : %A" perf

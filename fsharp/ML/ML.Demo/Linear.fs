// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
module Linear

open ML.Core.Readers
open ML.Core.Utils
open ML.Regressions.GLM
open ML.Regressions.LinearRegression

//open ML.Regressions.BatchGradientDescent
open ML.Regressions.SGD
open ML.Regressions.NAG
(*
open ML.Regressions.StochasticGradientDescent
open ML.Regressions.MiniBatchGradientDescent
open ML.Regressions.NesterovAcceleratedGradient
open ML.Regressions.AdagradGradientDescent
open ML.Regressions.AdadeltaGradientDescent
open ML.Regressions.AdadeltaAcceleratedGradientDescent
*)



open MathNet.Numerics.LinearAlgebra
open PerfUtil

open ML.Statistics.Regressions
open ML.Statistics.Charting

let linear() = 
        
    let _inputs, outputs = readCSV @"..\..\..\..\..\machine-learning-ex1\ex1\ex1data2.csv" false [|0..1|] 2    
    let outputs = vector outputs
    let inputs = matrix _inputs
    let inputs, normPrms = norm inputs

    let model = {
        Hypothesis = linearHyp
        Cost = linearMSECost
        Gradient = linearMSEGradient
    }

    let prms = {
        EpochNumber = 5000 // Epochs number
        MinErrorThreshold = 0.
    }

    let basicHyper = {
        Alpha = 0.01
    }

    let batchHyper = {
        Basic = basicHyper
        BatchSize = inputs.RowCount
    }

    let stochasticHyper = {
        Basic = basicHyper
        BatchSize = 1
    }

    let SGDHyper = {
        Basic = basicHyper
        BatchSize = 5
    }

    let NAGHyper = {        
        SGD = SGDHyper
        Gamma = 0.5
    }


    let mutable trainResults = [] 
    
    let perf = Benchmark.Run (fun () ->
        let train = SGD model prms batchHyper inputs outputs        
        trainResults <- ("batch",train)::trainResults
        printfn "batch result : %A" train
    )    
    printfn "batch perf : %A" perf
    
    let perf = Benchmark.Run (fun () ->
        let train = SGD model prms stochasticHyper inputs outputs
        trainResults <- ("stochastic", train)::trainResults
        printfn "stochastic result : %A" train
    )    
    printfn "stochastic perf : %A" perf

    let perf = Benchmark.Run (fun () ->
        let train = SGD model prms SGDHyper inputs outputs
        trainResults <- ("SGD", train)::trainResults
        printfn "miniBatch result : %A" train
    )    
    printfn "miniBatch perf : %A" perf

    let perf = Benchmark.Run (fun () ->
        let train = NAG model prms NAGHyper inputs outputs
        trainResults <- ("NAG", train)::trainResults
        printfn "NAG result : %A" train
    )    
    printfn "NAG perf : %A" perf
        
    trainResults
    |> List.sortBy (fun (_, res) -> res.Errors.[0])
    |> List.map (fun (label, res) -> (sprintf "%s : %f (%i)" label res.Errors.[0] res.Errors.Length), res.Errors |> List.mapi(fun i x -> (float i, x)))        
    |> showLines2

    

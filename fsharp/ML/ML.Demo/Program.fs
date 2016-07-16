// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.

open ML.Utils
open ML.Regressions.GLM
open ML.Regressions.LinearRegression
open ML.Regressions.BatchGradientDescent
open MathNet.Numerics.LinearAlgebra

[<EntryPoint>]
let main argv = 
    let inputs, outputs = readCSV @"C:\dev\ml\machine-learning-ex1\ex1\ex1data2.txt" false [|0..1|] 2    
    let outputs = vector outputs
    let inputs = norm  (matrix inputs)
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
    let train = batchGradientDescent model prms inputs outputs
    printfn "%A" train
    System.Console.ReadLine() |> ignore
    0 // return an integer exit code

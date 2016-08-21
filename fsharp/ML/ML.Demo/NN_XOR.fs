module NN_XOR

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

    let stochasticHyper : SGDHyperParams = {
        Alpha = 0.5
        BatchSize = 1
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

    let mutable trainResults = [] 

    let perf = Benchmark.Run (fun () ->
        let train = stochasticHyper |> SGDHyperParams |> gradientDescent glmModel prms x y
        trainResults <- ("batch",train)::trainResults
        printfn "batch result : %A" train
    )

    let acc = accuracy shape x y 

    trainResults
    |> List.sortBy (fun (_, res) -> res.Errors.[0])
    |> List.map (fun (label, res) -> (sprintf "%s %f : %f (%i)" label (acc res.Theta) res.Errors.[0] res.Errors.Length), res.Errors |> List.mapi(fun i x -> (float i, x)))        
    |> showLines2
    
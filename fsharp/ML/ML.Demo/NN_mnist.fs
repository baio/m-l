module NN_mnist

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

let nn_mnist() = 
           
    printfn "Start reading"             
    let inputs, outputs  = readCSV2 @"c:/dev/.data/mnist/mnist_train.csv" false [|1..784|] 0 5000

    //let inputs, outputs  = readCSV2 @"c:/dev/.data/nmist_1.csv" false [|1..400|] 0 5000
    //let inputs, outputs = readCSV2 @"c:/dev/.data/mnist/mnist_train_norm.csv" true [|1..784|] 0 5000

    printfn "Reading is done"             
    
    let outputs = outputs |> vector |> encodeOneHot 10 |> flatMx 
    let inputs = matrix inputs
    //let inputs1, normPrms1 = norm inputs
    //let inputs2, normPrms2 = norm3 inputs
    //let inputs, _ = norm inputs

    //printfn "%A %A" inputs.[]
       
    let model = {
        Cost = NNCost
        Gradient = NNGradient
    }

    let prms = {
        EpochNumber = 50
        ConvergeMode = ConvergeModeNone
    }       


    let shape = 
        {
            Layers = 
                [ 
                    NNFullLayerShape({ NodesNumber = 784; Activation = act }); 
                    NNFullLayerShape({ NodesNumber = 50; Activation = sigm }); 
                    //{ NodesNumber = 400; Activation = act }; 
                    //{ NodesNumber = 25; Activation = sigm }; 
                    NNFullLayerShape({ NodesNumber = 10; Activation = sigm }); 
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
        Alpha = 0.05
        BatchSize = 1
    }

    let batchHyper : SGDHyperParams = {
        Alpha = 0.5
        BatchSize = inputs.RowCount
    }

    let minbatchHyper : SGDHyperParams = {
        Alpha = 0.5
        BatchSize = inputs.RowCount
    }

    let NAGHyper : NAGHyperParams = {        
        Alpha = 0.01
        BatchSize = inputs.RowCount
        Gamma = 0.5
    }

    let AdagradHyper : AdagradHyperParams = {        
        Alpha = 0.01
        BatchSize = inputs.RowCount
        Epsilon = 1E-8
    }

    let AdadeltaHyper : AdadeltaHyperParams = {        
        BatchSize = inputs.RowCount
        Epsilon = 1E-8
        Rho = 0.6
    }

    ///

    let gd = gradientDescent glmModel prms inputs outputs

    let mutable trainResults = [] 


    
    let perf = Benchmark.Run (fun () ->
        let train = batchHyper |> SGDHyperParams |> gd
        trainResults <- ("batch",train)::trainResults
        printfn "batch result : %A" train
    )
    printfn "batch perf : %A" perf

    printfn "Start clalc"

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
        
    let acc = accuracy oneHot shape inputs outputs

    trainResults
    |> List.filter (fun (_, f) -> f.Errors.Length > 0)
    |> List.sortBy (fun (_, res) -> res.Errors.[0])
    |> List.map (fun (label, res) ->
         (sprintf "%s %f : %f (%i)" label (acc res.Theta) res.Errors.[0] res.Errors.Length), res.Errors 
         |> List.mapi(fun i x -> (float i, x))
    )        
    |> showLines2
    
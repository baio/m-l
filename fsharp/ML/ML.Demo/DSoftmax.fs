module DSoftmax

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

open ML.DGD.Types
open ML.DGD.DistributedGradientDescnt
open ML.DGD.SamplesStorage


open MathNet.Numerics.LinearAlgebra
open PerfUtil

open ML.Statistics.Regressions
open ML.Statistics.Charting

let DSoftmax() = 
       
    let model : GLMSoftmaxModel = {
        Base = { Cost = softmaxCost; Gradient = softmaxGradient }
        ClassesNumber = 3
    }

    let batchHyper : SGDHyperParams = {
        Alpha = 0.01
        BatchSize = 150
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

     //let inputs, outputs = readCSV @"..\..\..\..\..\machine-learning-ex2\ex2\ex2data1.txt" false [|0..1|] 2  
    //let inputs, outputs = readCSV @"..\..\..\..\..\data\iris.csv" true [|0..3|] 5
    //let inputs, outputs = readCSV2 @"c:/dev/.data/mnist/mnist_train.csv" false [|1..784|] 0 5000

    let samplesStoarge = {
            Location = SamplesStorageFile(@"..\..\..\..\..\data\iris.csv");
            Features = [0..3];
            Label = 5;
        }


    let dgdPrms = {    
        Model = GLMSoftmaxModel(model)
        HyperParams = SGDHyperParams(batchHyper)
        EpochNumber = 400
        //Samples storage
        SamplesStorage = samplesStoarge
        //Distributed batch size
        DistributedBatchSize = 1
        //How GDBatch get samples
        BatchSamples = BatchSamplesProvidedByCoordinator 
    }


    let mutable trainResults = [] 
    
    let perf = Benchmark.Run (fun () ->
        let train = DGD dgdPrms
        trainResults <- ("batch",train)::trainResults
        printfn "batch result : %A" train
    ) 
       
    printfn "batch perf : %A" perf
    
    let perf = Benchmark.Run (fun () ->
        let train = DGD { dgdPrms with HyperParams = SGDHyperParams(stochasticHyper) }
        trainResults <- ("stochastic", train)::trainResults
        printfn "stochastic result : %A" train
    )    
    printfn "stochastic perf : %A" perf

    let perf = Benchmark.Run (fun () ->
        let train = DGD { dgdPrms with HyperParams = SGDHyperParams(SGDHyper) }
        trainResults <- ("SGD", train)::trainResults
        printfn "miniBatch result : %A" train
    )    
    printfn "miniBatch perf : %A" perf

    let perf = Benchmark.Run (fun () ->
        let train = DGD { dgdPrms with HyperParams = NAGHyperParams(NAGHyper) }
        trainResults <- ("NAG", train)::trainResults
        printfn "NAG result : %A" train
    )    
    printfn "NAG perf : %A" perf
    
    
    let perf = Benchmark.Run (fun () ->
        let train = DGD { dgdPrms with HyperParams = AdagradHyperParams(AdagradHyper) }
        trainResults <- ("Adagrad", train)::trainResults
        printfn "Adagrad result : %A" train
    )    
    printfn "Adagrad perf : %A" perf
    
    let perf = Benchmark.Run (fun () ->
        let train = DGD { dgdPrms with HyperParams = AdadeltaHyperParams(AdadeltaHyper) }
        trainResults <- ("Adadelta", train)::trainResults
        printfn "Adadelta result : %A" train
    )    
    printfn "Adadelta perf : %A" perf
    
    let inputs, outputs = readCSV @"..\..\..\..\..\data\iris.csv" true [|0..3|] 5
    let outputs = vector outputs 
    let inputs = matrix inputs
    let inputs, normPrms = norm inputs
    let acc = accuracy model.ClassesNumber inputs outputs
        
    trainResults
    |> List.sortBy (fun (_, res) -> res.Errors.[0])
    |> List.map (fun (label, res) -> (sprintf "%s : %f %f (%i)" label (acc res.Theta)  res.Errors.[0] res.Errors.Length), res.Errors |> List.mapi(fun i x -> (float i, x)))        
    |> showLines2

    


module ML.NN.NN

open MathNet.Numerics.LinearAlgebra
open Nessos.Streams
open ML.Core.LinearAlgebra

open Types

type ActivationFun = { f: FVector -> FVector; f' : FVector -> FVector }

type LayerShape = {
    NodesNumber: int
    Activation: ActivationFun
}

type NNShape = {
    Layers: LayerShape list
}

type NNLayerType = 
    | Initial
    // Int number of nodes in input layer
    | Input of int 
    // Int number of nodes in input layer, thetas tail
    | Hidden of int * FVector * (FMatrix * ActivationFun)
    | Output of FMatrix * ActivationFun

// Given shape of NN and long vector of thetas, returns for each hidden layer : matrix of thetas + activation func - for each layer
// where rows from matrix containing one theta vector for each Node in the layer (including bias) 
let reshapeNN (shape: NNShape) (theta: FVector) : (FMatrix * ActivationFun) array = 

    let makeHidden (theta: FVector) pervLayerNodesNumber (layer: LayerShape) =
        // +1 for bias
        let layerThetasNumber = (pervLayerNodesNumber + 1) * layer.NodesNumber
        let mx = reshape (layer.NodesNumber, (pervLayerNodesNumber + 1)) theta.[..layerThetasNumber - 1]
        if theta.Count = layerThetasNumber then 
            // this is output layer
            Output(mx, layer.Activation)                            
        elif theta.Count > layerThetasNumber then
            Hidden(layer.NodesNumber, theta.[layerThetasNumber..],  (mx, layer.Activation))
        else 
            failwith "Number of thetas doesn't corresponds shape of NN"
            
    shape.Layers
    |> Stream.ofList
    |> Stream.scan (fun acc v ->        
        // calculate how many nodes was in pervious layer
        // if this is first layer then use same number of thetas as nodes number 
        match acc with
        | Initial ->
            Input(v.NodesNumber)
        | Input (nnumber) ->            
            makeHidden theta nnumber v
        | Hidden (nnumber, theta, _) ->
            makeHidden theta nnumber v
        | Output _ -> 
            failwith "Output must be latest layer"            
    ) Initial
    |> Stream.choose (fun res ->
        match res with
        | Hidden(_, _, res) -> 
            Some(res)
        | Output(a, b) -> 
            Some(a, b)
        | _ ->
            Option.None
    )
    |> Stream.toArray

type LayerForwardResult = {
    Weights: FMatrix
    Net : FVector
    //Out is not required in hidden layers
    Out : FVector
    Activation : ActivationFun
}

type LayerForward = 
    | Input of FVector
    | Hidden of LayerForwardResult

// returns a, z
let calcLayerForward (theta: FMatrix) (activation: ActivationFun) (inputs: FVector) = 
  let res1 = theta * (inputs |> appendOne) 
  res1 |> activation.f, res1
  
let forward2 (inputs: FVector) layers = 
    layers
    |> Array.scan (fun acc (th, act) ->
        let layerInputs = match acc with | Input (i) -> i | Hidden (l) -> l.Out
        let out, net = calcLayerForward th act layerInputs
        Hidden({Weights = th; Net = net; Out = out; Activation = act})
    ) (Input(inputs))
  
let forward (inputs: FVector) (shape: NNShape) (theta: FVector) = 
    let outs = 
        reshapeNN shape theta 
        |> forward2 inputs
    let res = outs.[outs.Length - 1]
    match res with | Hidden l -> l.Out | _ -> failwith "Last layer must be hidden"


type LayerBackwordResult = { Delta : FMatrix; Gradient : FMatrix }

type LayerBackword = 
    | LBNone
    | LBOutput of FVector // Delta of output layer
    | LBHidden of FMatrix

let backward (outputs: FVector) (inputs: FVector) (shape: NNShape) (theta: FVector) =
    
    let layers = reshapeNN shape theta
    
    let fwd = forward2 inputs layers
    
    Array.scanBack (fun v acc ->             
        match v with
        | Hidden l ->
            match acc with 
            | LBNone ->
                //ouput layer net
                let deltaOut = l.Activation.f' l.Net
                let deltaE = l.Out - outputs
                let deltaRes = deltaOut .* deltaE
                LBOutput(deltaRes)
            | LBOutput dOut ->
                //hidden layer (last)                
                let outMx = [l.Out] |> DenseMatrix.ofRowSeq |> appendOnes |>  repmatRow dOut.Count
                let dOutMx = [dOut] |> DenseMatrix.ofColumnSeq |> repmatCol outMx.ColumnCount
                let dEtotal_dw = outMx .* dOutMx
                LBHidden dEtotal_dw
            | LBHidden delta ->
                LBNone                                                        
        | Input _ -> 
            LBNone
    ) fwd LBNone
    |> Array.choose (function
        | LBHidden l -> Some(l)
        | _ -> None
    )



    //https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/                
    //http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/



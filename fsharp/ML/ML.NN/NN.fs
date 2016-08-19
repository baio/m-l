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
    //same as `W`
    Weights: FMatrix
    //same as `z`
    Net : FVector
    //same as `a`
    Out : FVector
    //Same as `f'`
    Activation : ActivationFun
}

//ForwardResult
type LayerForward = 
    | Input of FVector // inputs
    | Hidden of LayerForwardResult

// returns a, z
let calcLayerForward (theta: FMatrix) (activation: ActivationFun) (inputs: FVector) = 
  let z = theta * (appendOne inputs) 
  (activation.f z), z
  
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

//BackpropResult
type LayerBackword = 
    | LBNone
    | LBOutput of FMatrix * FVector // Weights of layer * Delta of output layer 
    | LBHidden of FMatrix * FVector * FMatrix // weights of layer * Delta * Gradient
    | LBInput of FMatrix // gradient

let private caclGrads a_l delta_l_1 = 
    let a_l_mx = [a_l] |> DenseMatrix.ofRowSeq |> appendOnes
    let delta_l_1_mx = [delta_l_1] |> DenseMatrix.ofColumnSeq
    delta_l_1_mx * a_l_mx

let private caclHiddenDelta fwdLayer (weights_l_1: FMatrix) delta_l_1 = 
    let delta_out = fwdLayer.Activation.f' fwdLayer.Net
    let delta_l_1_mx = [delta_l_1] |> DenseMatrix.ofRowSeq
    let delta_e = (delta_l_1_mx * weights_l_1.RemoveColumn(0)).Row(0)
    delta_out .* delta_e
    
let backprop (outputs: FVector) (inputs: FVector) (shape: NNShape) (theta: FVector) =
    
    //https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/                
    //http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/

    let layers = reshapeNN shape theta
    
    let fwd = forward2 inputs layers
    
    Array.scanBack (fun v acc ->             
        match v, acc with
        | (Hidden l, LBNone) ->
            //ouput layer net
            let deltaOut = l.Activation.f' l.Net
            let deltaE = l.Out - outputs
            let delta = deltaOut .* deltaE
            LBOutput(l.Weights, delta)
        | (Hidden l, LBOutput (weights_l_1, delta_l_1)) ->
            //last hidden layer (n_l - 1)
            let delta_l = caclHiddenDelta l weights_l_1 delta_l_1     
            let grad_l = caclGrads l.Out delta_l_1
            LBHidden(l.Weights, delta_l, grad_l)
        | (Hidden l, LBHidden (weights_l_1, delta_l_1, _)) ->
            //gradient for hidden layer (n_l - 2...)
            let delta = caclHiddenDelta l weights_l_1 delta_l_1
            let grad = caclGrads l.Out delta_l_1
            LBHidden(l.Weights, delta, grad)
        | (Input inputs, LBHidden (weights, delta_l_1, _)) ->
            // calc gradient for first hidden layer (n_1)
            let grad = caclGrads inputs delta_l_1
            LBInput(grad)
        | (Input inputs, LBOutput (_, delta_l_1)) ->
            // one layer network case (n_1)
            let grad = caclGrads inputs delta_l_1
            LBInput(grad)
        | _ ->             
           failwith "not supported"
    ) fwd LBNone
    |> Array.choose (function
        | LBHidden(_, _, grad) -> Some(grad)
        | LBInput(grad) -> Some(grad)
        | _ -> None
    )


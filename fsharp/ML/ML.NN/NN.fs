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
    | None    
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
        | None ->
            Input(v.NodesNumber)
        | Input (nnumber) ->            
            makeHidden theta nnumber v
        | Hidden (nnumber, theta, _) ->
            makeHidden theta nnumber v
        | Output _ -> 
            failwith "Output must be latest layer"            
    ) NNLayerType.None
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

let calcLayerForward (theta: FMatrix) (activation: ActivationFun) (inputs: FVector) : FVector = 
  let res = theta * (inputs |> appendOne) |> activation.f
  res

let forward2 (inputs: FVector) layers = 
    layers
    |> Array.fold (fun acc (th, act) ->
        calcLayerForward th act acc
    ) inputs
  
let forward (inputs: FVector) (shape: NNShape) (theta: FVector) = 
    reshapeNN shape theta
    |> forward2 inputs

(*
let backProp (outputs: FVector) (inputs: FVector) (shape: NNShape) (theta: FVector) =
    let layers = reshapeNN shape theta
    let res = calcNN2 inputs layers
    let deltas = layers |> Array.foldBack (fun acc (th, act) ->
        match acc with
        | None ->
            // first layer
            Some(-1 * (res - outputs) * act.f'(th.Head() |> Vector.sum))
        | Some delta ->            
                   
            
    ) None
    ()
*)


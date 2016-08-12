//http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/
module Types

open Nessos.Streams
open MathNet.Numerics.LinearAlgebra

open ML.Core.LinearAlgebra

type FVector = float Vector

type FMatrix = float Matrix

type ActivationFun = FVector -> FVector

type LayerShape = {
    NodesNumber: int
    Activation: ActivationFun
}

type NNShape = {
    Layers: LayerShape list
}

let calcLayer (theta: FMatrix) (activation: ActivationFun) (inputs: FVector) : FVector = 
    theta * inputs |> activation

// Given shape of NN and long vector of thetas, returns for each layer : matrix of thetas + activation func - for each layer
// where rows from matrix containing one theta vector for each Node in the layer (including bias) 
let reshapeNN (shape: NNShape) (theta: FVector) : (FMatrix * ActivationFun) array = 
    let _input = shape.Layers.[0]
    let _hidden = shape.Layers.[1..]
    let input = theta.[0.._input.NodesNumber + 1]
    shape.Layers
    |> Stream.ofList
    |> Stream.scan (fun acc v ->
        // calculate how many nodes was in pervious layer
        //if this is first layer then use same number of thetas as nodes number 
        match acc with
        | None ->
            let mx = reshape (v.NodesNumber, 1) theta.[..v.NodesNumber]
            Some (mx, v.Activation, theta.[v.NodesNumber..])
        | Some (prevTheta, _, theta) ->            
            let pervLayerNodesNumber = prevTheta.ColumnCount           
            let layerThetasNumber = v.NodesNumber * pervLayerNodesNumber
            let mx = reshape (v.NodesNumber, pervLayerNodesNumber) theta.[..layerThetasNumber]
            Some (mx, v.Activation, theta.[layerThetasNumber..])
            
    ) None
    |> Stream.choose (fun res ->
        match res with
        | Some (a, b, _) -> Some(a, b)
        | None -> None
    )
    |> Stream.toArray

let calcNN (inputs: FVector) (shape: NNShape) (theta: FVector) = 
    let layers = reshapeNN shape theta
    let firstLayer, _ = layers.[0]
    let inputs = firstLayer.Column(0)

    layers.[1..]    
    |> Array.fold (fun acc (th, act) ->
        calcLayer th act acc
    ) inputs

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

// Notation notes :
// Z, z, Net, N - layer net (b + x1*w1 + x2*w2 + ..)
// A, a, Out, O - layer output (f' (Z))
// partials : ∂ the same as Δ; 𝟃E/𝟃A same as ΔE_ΔA
// Weight, W, w, Theta
// δᴸᴾ = Delta in layer L + 1, same as δᴸ⁺¹, same for other symbols
// gradients : Δᴸ, Δ, ∇

//BackpropResult
type LayerBackword =
    | LBNone
    | LBOutput of FMatrix * FVector // Weights of layer * Delta of output layer
    | LBHidden of FMatrix * FVector * FMatrix // weights of layer * Delta * Gradient
    | LBInput of FMatrix // gradient

let private caclGrads aᴸ δᴸᴾ =
    let aᴸ_mx = [aᴸ] |> DenseMatrix.ofRowSeq |> appendOnes
    let δᴸᴾ_mx = [δᴸᴾ] |> DenseMatrix.ofColumnSeq
    δᴸᴾ_mx * aᴸ_mx

let private caclHiddenDelta fwdLayer (wᴸᴾ: FMatrix) δᴸᴾ =
    let δᴸᴾ_mx = [δᴸᴾ] |> DenseMatrix.ofRowSeq
    let ΔE_ΔA = (δᴸᴾ_mx * wᴸᴾ.RemoveColumn(0)).Row(0)
    let ΔA_ΔN = fwdLayer.Activation.f' fwdLayer.Net
    ΔE_ΔA .* ΔA_ΔN

let backprop (outputs: FVector) (inputs: FVector) (shape: NNShape) (theta: FVector) =

    //https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    //http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/

    // Delta for layers calculated as such :
    // δᴸ = 𝟃E/𝟃A * 𝟃A/𝟃N
    // =========
    // 𝟃E/𝟃A
    // how much does the total error change with respect to the output?
    // calculated differntly for output and hiddent layers (see readme)
    // =========
    // 𝟃A/𝟃N
    // how much does the output change with respect to its total net input
    // calculated similary for all layers f'(Net)
    // Delta not calculated for input layer and biases,
    // only for nodes which have ingoing connections,
    // since these deltas will be needed to calculate gradients for these connections

    // Gradients for each node calculated as such (using chain rule)
    // ∇ = 𝟃E/𝟃A * 𝟃A/𝟃N * 𝟃N/𝟃W
    // where 𝟃E/𝟃A * 𝟃A/𝟃N is the delta from previous layer (δᴸ⁺¹)
    // Hence ∇ = δᴸ⁺¹ * 𝟃N/𝟃W
    // Where 𝟃N/𝟃W = aᴸ (see why in readme)
    // Indeed ∇ = δᴸ⁺¹ * aᴸ (for input layer aᴸ = inputs)

    let layers = reshapeNN shape theta

    let fwd = forward2 inputs layers

    Array.scanBack (fun v acc ->
        match v, acc with
        | (Hidden l, LBNone) ->
            // ouput layer net
            // this is first calculated layer in backprop alghoritm
            let ΔE_ΔA = l.Out - outputs
            let ΔA_ΔN = l.Activation.f' l.Net
            let δᴸ = ΔE_ΔA .* ΔA_ΔN
            LBOutput(l.Weights, δᴸ)
        | (Hidden l, LBOutput (wᴸᴾ, δᴸᴾ)) ->
            //last hidden layer (n_l - 1), right before outputs
            let δᴸ = caclHiddenDelta l wᴸᴾ δᴸᴾ
            let Δᴸ = caclGrads l.Out δᴸᴾ
            LBHidden(l.Weights, δᴸ, Δᴸ)
        | (Hidden l, LBHidden (wᴸᴾ, δᴸᴾ, _)) ->
            //gradient for hidden layer (n_l - 2...)
            let δᴸ = caclHiddenDelta l wᴸᴾ δᴸᴾ
            let Δᴸ = caclGrads l.Out δᴸᴾ
            LBHidden(l.Weights, δᴸ, Δᴸ)
        | (Input inputs, LBHidden (_, δᴸᴾ, _)) ->
            // calc gradient for first hidden layer (n_1)
            let Δᴸ = caclGrads inputs δᴸᴾ
            LBInput(Δᴸ)
        | (Input inputs, LBOutput (_, δᴸᴾ)) ->
            // one layer network case (n_1)
            let Δᴸ = caclGrads inputs δᴸᴾ
            LBInput(Δᴸ)
        | _ ->
           failwith "not supported"
    ) fwd LBNone
    |> Array.choose (function
        | LBHidden(_, _, grad) -> Some(grad)
        | LBInput(grad) -> Some(grad)
        | _ -> None
    )


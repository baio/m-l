module ML.NN

open MathNet.Numerics.LinearAlgebra
open Nessos.Streams
open ML.Core.LinearAlgebra

///////////////////////// NN Shape

type ActivationFun = { f: FVector -> FVector; f' : FVector -> FVector }

type NNLayerShape = {
    NodesNumber: int
    Activation: ActivationFun
}

type NNShape = {
        Layers: NNLayerShape list
} with 
    member 
        this.thetasCount() = 
            (0, 0)
            |> List.foldBack(fun layer (totalLinksCount, prevLayerNodesCount) ->
                let layerLinksCount = (layer.NodesNumber + 1) * prevLayerNodesCount
                (totalLinksCount + layerLinksCount), layer.NodesNumber
            ) this.Layers
            |> fst

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

    let makeHidden (theta: FVector) pervLayerNodesNumber (layer: NNLayerShape) =
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

/////////////////////////////////////////Forward

type ForwardInputLayerResult = {
    Inputs: FMatrix
}

type ForwardHiddenLayerResult = {
    //same as `W`
    Weights: FMatrix
    //same as `z`
    Net : FMatrix
    //same as `a`
    Out : FMatrix
    //Same as `f'`
    Activation : ActivationFun
}

//ForwardResult
type ForwardResult =
    | ForwardResultInput of ForwardInputLayerResult // inputs
    | ForwardResultHidden of ForwardHiddenLayerResult

// returns a, z
let calcLayerForward (theta: FMatrix) (activation: ActivationFun) (inputs: FMatrix) =
  let z = (appendOnes inputs).TransposeAndMultiply(theta)
  // TODO activation must recieve matrix
  (z |> mapRows activation.f), z

let forward2 (inputs: FMatrix) layers =
    layers
    |> Array.scan (fun acc (th, act) ->
        let layerInputs =
            match acc with
            | ForwardResultInput ({Inputs = inputs}) -> inputs
            | ForwardResultHidden({Out = inputs}) -> inputs
        let out, net = calcLayerForward th act layerInputs
        ForwardResultHidden({ Weights = th; Net = net; Out = out; Activation = act })
    ) (ForwardResultInput({ Inputs = inputs }))

let forward (inputs: FMatrix) (shape: NNShape) (theta: FVector) =
    reshapeNN shape theta |> (forward2 inputs)

let forwardOutput (inputs: FMatrix) (shape: NNShape) (theta: FVector) =
    let fwds = forward inputs shape theta
    match fwds.[fwds.Length - 1] with 
    | ForwardResultHidden {Out = res} -> res
    | _ -> failwith "Output layer must be hidden"

///////////////////////// Backprop

// Notation notes :
// Z, z, Net, N - layer net (b + x1*w1 + x2*w2 + ..)
// A, a, Out, O - layer output (f' (Z))
// partials : ∂ the same as Δ; 𝟃E/𝟃A same as ΔE_ΔA
// Weight, W, w, Theta
// δᴸᴾ = Delta in layer L + 1, same as δᴸ⁺¹, same for other symbols
// gradients : Δᴸ, Δ, ∇


type BackpropOutputLayerResult = {
    Weights : FMatrix
    Delta : FVector
}

type BackpropHiddenLayerResult = {
    Weights : FMatrix
    Gradient : FMatrix
    Delta : FVector
}

type BackpropInputLayerResult = {
    Gradient : FMatrix
}

//BackpropResult
type BackpropResult =
    | BackpropResultNone
    | BackpropResultOutput of BackpropOutputLayerResult
    | BackpropResultHidden of BackpropHiddenLayerResult
    | BackpropResultInput of BackpropInputLayerResult

let private caclGrads aᴸ δᴸᴾ =
    let aᴸ_mx = [aᴸ] |> DenseMatrix.ofRowSeq |> appendOnes
    let δᴸᴾ_mx = [δᴸᴾ] |> DenseMatrix.ofColumnSeq
    δᴸᴾ_mx * aᴸ_mx

let private caclHiddenDelta fwdLayer (wᴸᴾ: FMatrix) δᴸᴾ =
    let δᴸᴾ_mx = [δᴸᴾ] |> DenseMatrix.ofRowSeq
    let ΔE_ΔA = (δᴸᴾ_mx * wᴸᴾ.RemoveColumn(0)).Row(0)
    let ΔA_ΔN = fwdLayer.Activation.f' (fwdLayer.Net.Row(0))
    ΔE_ΔA .* ΔA_ΔN

//TODO : inputs must already contain bias (check FBiasVector)
let private _backprop (outputs: FVector) (inputs: FVector) (shape: NNShape) (theta: FVector) =

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
    // =========
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

    let fwd = forward2 ([inputs] |> DenseMatrix.ofRowSeq) layers

    Array.scanBack (fun v acc ->
        match v, acc with
        | (ForwardResultHidden(l), BackpropResultNone) ->
            // ouput layer net
            // this is first calculated layer in backprop alghoritm
            let ΔE_ΔA = l.Out.Row(0) - outputs
            let ΔA_ΔN = l.Activation.f' (l.Net.Row(0))
            let δᴸ = ΔE_ΔA .* ΔA_ΔN
            BackpropResultOutput({ Weights = l.Weights; Delta = δᴸ })
        | (ForwardResultHidden(l), BackpropResultOutput({Weights = wᴸᴾ; Delta = δᴸᴾ})) 
        | (ForwardResultHidden(l), BackpropResultHidden ({Weights = wᴸᴾ; Delta = δᴸᴾ})) ->
            //last hidden layer (n_l - 1), right before outputs OR for hidden layer (n_l - 2...)
            let δᴸ = caclHiddenDelta l wᴸᴾ δᴸᴾ
            let Δᴸ = caclGrads (l.Out.Row(0)) δᴸᴾ
            BackpropResultHidden({ Weights = l.Weights; Delta =  δᴸ; Gradient = Δᴸ })                    
        | (ForwardResultInput(l), BackpropResultHidden({Delta =  δᴸᴾ}))             
        | (ForwardResultInput(l), BackpropResultOutput ({Delta = δᴸᴾ})) ->
            // calc gradient for first hidden layer (n_1) OR one layer network case (n_1)
            let Δᴸ = caclGrads (l.Inputs.Row(0)) δᴸᴾ
            BackpropResultInput({ Gradient = Δᴸ})
        | _ ->
           failwith "not supported"
    ) fwd BackpropResultNone
    |> Array.choose (function
        | BackpropResultHidden({ Gradient = Δ }) -> Some(Δ)
        | BackpropResultInput({ Gradient = Δ }) -> Some(Δ)
        | _ -> None
    )

let backprop (y: FVector) (x: FMatrix) (shape: NNShape) (theta: FVector) =
    //TODO : forward require inputs without bias
    let thetasCount = shape.thetasCount()
    let ys = chunkOutputs x.RowCount y
    let gradSum = 
        x.EnumerateRows()
        |> Stream.ofSeq
        |> Stream.mapi(fun i f ->
            _backprop (ys.Row i) f shape theta |> flatMxs
        )
        |> Stream.fold (fun acc v -> acc + v) (zeros thetasCount)
    gradSum / float x.RowCount

////

let NNCost _ (x : float Matrix) (y : float Vector) (theta: float Vector)  =     
    ((x * theta - y).PointwisePower(2.) |> Vector.sum) / (2. * float x.RowCount) 
        
let NNGradient _ (x : float Matrix) (y : float Vector) (theta: float Vector) =
    x.Transpose() * (x * theta - y) / float x.RowCount

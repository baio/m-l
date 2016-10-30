module ML.NN

open MathNet.Numerics.LinearAlgebra
open Nessos.Streams
open ML.Core.LinearAlgebra
open ML.Core.Utils

///////////////////////// NN Shape

type ActivationFun = { f: FVector -> FVector; f' : FVector -> FVector }

(**
    ## Define fully connected layer
**)
type NNFullLayerShape = {
    NodesNumber: int
    Activation: ActivationFun
}

(**
    ## Define embedded layer (dense representation)
    Nodes on this layer [j1, j2, j3, j4]
    Nodes from pervious layer [i1, i2, i3, i4, i5, i6]         
    BlocksNumber = 2
    NodesInBlockNumber = 2
    Connections:
    [i1, i2, i3] fully connected to [j1, j2]
    [i4, i5, i6] fully connected to [j3, j4]
**)
type NNEmbedLayerShape = {
    BlocksNumber: int
    NodesInBlockNumber: int
    Activation: ActivationFun
}

type NNLayerShape = 
    | NNFullLayerShape of NNFullLayerShape
    | NNEmbedLayerShape of NNEmbedLayerShape
with
    
    member
        (**
            Get total number of weights (with biases)
        **)
        this.NodesNumber with get() =
            match this with
                | NNFullLayerShape l -> l.NodesNumber
                | NNEmbedLayerShape l -> l.BlocksNumber * l.NodesInBlockNumber
    
    member
        this.Activation with get() =
            match this with
                | NNFullLayerShape l -> l.Activation
                | NNEmbedLayerShape l -> l.Activation
                
                

type NNShape = {
        Layers: NNLayerShape list
} with
    member
        (**
            Get total number of weights (with biases) 
        **)
        this.ThetasCount() =
            (0, 0)
            |> List.foldBack(fun (layer: NNLayerShape) (totalLinksCount, prevLayerNodesCount) ->
                let layerLinksCount = (layer.NodesNumber + 1) * prevLayerNodesCount
                (totalLinksCount + layerLinksCount), layer.NodesNumber
            ) this.Layers
            |> fst

type NNLayer = {
    Thetas: FMatrix list
    Activation: ActivationFun
}

/////////////////////////////////////// Reshape

type NNLayerReshapeInput = {
    NodesNumber: int
}

type NNLayerReshapeHidden = {
    NodesNumber: int
    Thetas: FMatrix list
    ThetasTail: FVector
    Activation: ActivationFun
}

type NNLayerReshape =
    | NNLayerReshapeNone
    | NNLayerReshapeInput of NNLayerReshapeInput
    | NNLayerReshapeHidden of NNLayerReshapeHidden
    | NNLayerReshapeOutput of NNLayer

let private createReshapeLayer (theta: FVector) nodesNumber layerThetasNumber thetas activation = 
    if theta.Count = layerThetasNumber then
        // this is output layer
        NNLayerReshapeOutput({Thetas = thetas; Activation = activation})
    elif theta.Count > layerThetasNumber then
        NNLayerReshapeHidden(
            {
                NodesNumber = nodesNumber;
                ThetasTail = theta.[layerThetasNumber..];
                Thetas = thetas;
                Activation =  activation
            })
    else
        failwith "Number of thetas doesn't corresponds shape of NN"


let makeHidden (theta: FVector) pervLayerNodesNumber (layer: NNLayerShape) =
    match layer with
        | NNFullLayerShape layer ->
            // +1 for bias
            let layerThetasNumber = (pervLayerNodesNumber + 1) * layer.NodesNumber
            let mx = reshape (layer.NodesNumber, (pervLayerNodesNumber + 1)) theta.[..layerThetasNumber - 1]
            createReshapeLayer theta layer.NodesNumber layerThetasNumber [mx] layer.Activation
        | NNEmbedLayerShape layer ->
            let thetas = 
                List.init layer.BlocksNumber (fun i -> 
                    if pervLayerNodesNumber % layer.BlocksNumber <> 0 then 
                        failwith "NodesNumber in pervious layer must be devidaed by BlocksNumber as integer"                        
                    // no bias
                    let pervLayerNodesInBlockNumber = pervLayerNodesNumber / layer.BlocksNumber
                    let thetasNumber = pervLayerNodesInBlockNumber * layer.NodesInBlockNumber
                    let thetaIndexFrom = i * thetasNumber
                    let thetaIndexTo = thetaIndexFrom + thetasNumber - 1
                    reshape (layer.NodesInBlockNumber, pervLayerNodesInBlockNumber) theta.[thetaIndexFrom..thetaIndexTo]
                ) 
            let layerThetasNumber = thetas |> List.map (fun m -> m.RowCount * m.ColumnCount) |>List.sum
            createReshapeLayer theta (layer.BlocksNumber * layer.NodesInBlockNumber) layerThetasNumber thetas layer.Activation

(**
    ## Convert Shaped Network into flattened vector representation 
**)
let reshapeNN (shape: NNShape) (theta: FVector)  =

    shape.Layers
    |> Stream.ofList
    |> Stream.scan (fun st l ->
        // calculate how many nodes was in pervious layer
        match st with
        | NNLayerReshapeNone ->
            // if this is first layer then use same number of thetas as nodes number
            NNLayerReshapeInput({ NodesNumber = l.NodesNumber })
        | NNLayerReshapeInput ({ NodesNumber = pervNodesNumber }) ->
            makeHidden theta pervNodesNumber l
        | NNLayerReshapeHidden ({ NodesNumber = pervNodesNumber; ThetasTail = thetasTail }) ->
            makeHidden thetasTail pervNodesNumber l
        | NNLayerReshapeOutput _ ->
            failwith "Output must be latest layer"
    ) NNLayerReshapeNone
    |> Stream.choose (function
        | NNLayerReshapeHidden({Thetas = thetas; Activation = activation})
        | NNLayerReshapeOutput({Thetas = thetas; Activation = activation}) ->
            Some({Thetas = thetas; Activation = activation} : NNLayer)
        | _ ->
            None
    )
    |> Stream.toArray

///////////////////////////////////////// Forward

type ForwardInputLayerResult = {
    Inputs: FMatrix
}

type ForwardHiddenLayerResult = {
    //same as `W`
    Weights: FMatrix list
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

(**
   ## Given weights, activation function and inputs calculates activated and preactivated outputs

   ### Returns tulpe
   + First ulpe element
    contains Matrix where each row corresponds to each input and each column - activated output of this input calculated for each layer's node
   + Second tulpe element 
    contains Matrix where each row corresponds to each input and each column - preactivated output of this input calculated for each layer's node
**)
let calcLayerForward useBias (theta: FMatrix) (activation: ActivationFun) (inputs: FMatrix) =
  //add bias to inputs if required
  let binputs = iif useBias (appendOnes inputs) inputs
  // precativated output
  let z = binputs.TransposeAndMultiply(theta)
  // activated + prectivated output
  (z |> mapRows activation.f), z

(**
   ### Given inputs and network layers claculate forward propagatin results
**)
let forward2 (inputs: FMatrix) layers =
    //    
    layers
    |> Array.scan (fun st ({Thetas = layerTheta; Activation = layerActivation} : NNLayer) ->
        let layerInputs =
            match st with
            | ForwardResultInput ({Inputs = inputs}) -> inputs
            | ForwardResultHidden({Out = inputs}) -> inputs
        match layerTheta with
        | [theta] ->
            // fully connected layer
            let out, net = calcLayerForward true theta layerActivation layerInputs
            ForwardResultHidden({ Weights = [theta]; Net = net; Out = out; Activation = layerActivation })
        | thetas ->
            // embed layer
            let chunkedInputs = layerInputs |> chunkColumns thetas.Length
            thetas
            |> List.fold (fun (i, st) theta ->                
                let blockInputs = chunkedInputs.[i]
                let lout, lnet = calcLayerForward false theta layerActivation blockInputs
                match st with
                | Some (out, net) -> i + 1, Some((appendColumns out lout), (appendColumns net lnet))
                | None -> 1, Some(lout, lnet)
            ) (0, None)
            |> function
                | _, Some(out, net) ->
                    ForwardResultHidden({ Weights = thetas; Net = net; Out = out; Activation = layerActivation })
                | _, None ->
                    failwith "Layer data inconsistent"

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
    Weights : FMatrix list
    Delta : FMatrix // delats for each sample in batch
}

type BackpropHiddenLayerResult = {
    Weights : FMatrix list
    Gradient : FMatrix
    Delta : FMatrix // delats for each sample in batch
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

let private caclGrads (aᴸ: FMatrix) (δᴸᴾ: FMatrix) =
    let aᴸ = aᴸ |> appendOnes
    let δᴸᴾ = δᴸᴾ.Transpose()
    δᴸᴾ * aᴸ

let private caclHiddenDelta fwdLayer (wᴸᴾ: FMatrix) δᴸᴾ =
    let ΔE_ΔA = δᴸᴾ * wᴸᴾ.RemoveColumn(0)
    let ΔA_ΔN = fwdLayer.Net |> mapRows fwdLayer.Activation.f'
    ΔE_ΔA .* ΔA_ΔN

//TODO : inputs must already contain bias (check FBiasVector)
let private _backprop (Y: FMatrix) (X: FMatrix) (shape: NNShape) (theta: FVector) =

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

    let fwd = forward2 X layers

    Array.scanBack (fun v acc ->
        match v, acc with
        | (ForwardResultHidden(l), BackpropResultNone) ->
            // ouput layer
            // this is first calculated layer in backprop alghoritm
            // just calc δᴸ
            let ΔE_ΔA = l.Out - Y
            let ΔA_ΔN =  l.Net |> mapRows l.Activation.f'
            let δᴸ = ΔE_ΔA .* ΔA_ΔN
            BackpropResultOutput({ Weights = l.Weights; Delta = δᴸ })
        | (ForwardResultHidden(l), BackpropResultOutput({Weights = wᴸᴾ; Delta = δᴸᴾ}))
        | (ForwardResultHidden(l), BackpropResultHidden ({Weights = wᴸᴾ; Delta = δᴸᴾ})) ->
            //hidden layers (h_l-1, h_l-2...)
            match wᴸᴾ with
            | [wᴸᴾ] -> 
                let δᴸ = caclHiddenDelta l wᴸᴾ δᴸᴾ
                let Δᴸ = caclGrads l.Out δᴸᴾ
                BackpropResultHidden({ Weights = l.Weights; Delta =  δᴸ; Gradient = Δᴸ })
            | _ -> failwith "not implemented"            
        | (ForwardResultInput(l), BackpropResultHidden({Delta =  δᴸᴾ; Weights = wᴸᴾ}))
        | (ForwardResultInput(l), BackpropResultOutput ({Delta = δᴸᴾ; Weights = wᴸᴾ})) ->
            // calc gradient for first hidden layer (n_1)
            match wᴸᴾ with
            | [wᴸᴾ] -> 
                let Δᴸ = caclGrads l.Inputs δᴸᴾ
                BackpropResultInput({ Gradient = Δᴸ})
            | _ -> 
                let blocksNumber = wᴸᴾ |> List.length
                let chunkedDeltas = δᴸᴾ |> chunkColumns2 blocksNumber
                let chunkedInputs = l.Inputs |> chunkColumns2 blocksNumber
                let Δᴸ = 
                    Array.map2 caclGrads chunkedDeltas chunkedInputs
                    |> foldByColumns (fun vecs ->                    
                        let vecsLength = vecs |> Seq.length |> float
                        let vecLength = vecs |> Seq.item 0 |> Vector.length
                        vecs
                        |> Seq.fold (+) (zeros vecLength)
                        |> fun x -> x / vecsLength
                    ) 
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
    let Y = chunkOutputs x.RowCount y
    // grads for each sample (per layer)
    //TODO: improve concat withou mapping to array if possible
    let bp = _backprop Y x shape theta
    bp
    |> Array.collect(fun mx ->
        // avg weighted grads per layer
        mx |> Matrix.map(fun m -> m / float x.RowCount) |> flatMx |> Vector.toArray
    )
    |> DenseVector.ofArray

//////////////// Theta initailization

let private calcInitialLayerTheta useBias L_in L_out =
    let epsilon = System.Math.Sqrt(6.) / System.Math.Sqrt(float L_in + float L_out)
    let r = rndvec ((L_in + (iif useBias 1 0)) * L_out)
    r * 2. * epsilon - epsilon |> Vector.toList

//Get initial theta randomized
let getInitialTheta (shape: NNShape) =
    shape.Layers
    |> List.mapFold (fun st l_out  ->
        let res =
            match st with
            | None -> []
            | Some l_in -> 
                match l_out with
                | NNEmbedLayerShape l ->                    
                    calcInitialLayerTheta false (l_in / l.BlocksNumber) l.NodesInBlockNumber 
                    |> List.replicate l.BlocksNumber
                    |> List.collect (fun f -> f)
                | _ ->
                    calcInitialLayerTheta true l_in l_out.NodesNumber 
        res, Some l_out.NodesNumber
    ) None
    |> fst
    |> List.concat
    |> vector


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
    Nodes in this layer [j1, j2, j3, j4]
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
    Shape: NNLayerShape
    Thetas: FMatrix
}

/////////////////////////////////////// Reshape

type NNLayerReshapeInput = {
    NodesNumber: int
}

type NNLayerReshapeHidden = {
    Thetas: FMatrix
    ThetasTail: FVector option
    Shape: NNLayerShape
}

type NNLayerReshape =
    | NNLayerReshapeNone
    | NNLayerReshapeInput of NNLayerReshapeInput
    | NNLayerReshapeHidden of NNLayerReshapeHidden

let makeHidden (theta: FVector) pervLayerNodesNumber (layerShape: NNLayerShape) =
    match layerShape with
        | NNFullLayerShape shape ->
            // +1 for bias
            let layerThetasNumber = (pervLayerNodesNumber + 1) * shape.NodesNumber
            let mx = reshape (shape.NodesNumber, (pervLayerNodesNumber + 1)) theta.[..layerThetasNumber - 1]
            { Thetas = mx; ThetasTail = ifopt (theta.Count > layerThetasNumber) (fun () -> theta.[layerThetasNumber..]); Shape = layerShape }
        | NNEmbedLayerShape shape ->
            if pervLayerNodesNumber % shape.BlocksNumber <> 0 then
                failwith "NodesNumber in pervious layer must be devidaed by BlocksNumber as integer"
            // no bias
            let layerThetasNumber = (pervLayerNodesNumber / shape.BlocksNumber) * shape.NodesInBlockNumber
            let mx = reshape (shape.NodesInBlockNumber, pervLayerNodesNumber) theta.[..layerThetasNumber - 1]
            { Thetas = mx; ThetasTail = Some(theta.[layerThetasNumber..]); Shape = layerShape }

(**
    ## Convert Shaped Network into flattened vector representation
**)
let reshapeNN (shape: NNShape) (theta: FVector)  =

    shape.Layers
    |> Stream.ofList
    //TODO: reduce
    |> Stream.scan (fun st l ->
        // calculate how many nodes was in pervious layer
        match st with
        | NNLayerReshapeNone ->
            NNLayerReshapeInput({ NodesNumber = l.NodesNumber })
        | NNLayerReshapeInput ({ NodesNumber = pervNodesNumber }) ->
            NNLayerReshapeHidden(makeHidden theta pervNodesNumber l)
        | NNLayerReshapeHidden {ThetasTail = Some(thetasTail); Shape = shape} ->
            NNLayerReshapeHidden(makeHidden thetasTail shape.NodesNumber l)
    ) NNLayerReshapeNone
    |> Stream.choose (fun l ->
        match l with
        | NNLayerReshapeHidden { Thetas = thetas; Shape = shape } -> 
            Some({Thetas = thetas; Shape = shape} : NNLayer)
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
    Thetas: FMatrix
    //same as `z`
    Net : FMatrix
    //same as `a`
    Out : FMatrix
    //Same as `f'`
    Shape: NNLayerShape
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
    |> Array.scan (fun st ({Thetas = thetas; Shape = shape} : NNLayer) ->
        let layerInputs =
            match st with
            | ForwardResultInput ({Inputs = inputs}) -> inputs
            | ForwardResultHidden({Out = inputs}) -> inputs
        match shape with
        | NNFullLayerShape(layerShape) ->
            // fully connected layer
            let out, net = calcLayerForward true thetas shape.Activation layerInputs
            ForwardResultHidden({ Thetas = thetas; Net = net; Out = out; Shape = shape })
        | NNEmbedLayerShape(layerShape) ->
            // embed layer
            let chunkedInputs = layerInputs |> chunkColumns layerShape.BlocksNumber
            let lout, lnet = (List.init layerShape.BlocksNumber (fun i ->
                let blockInputs = chunkedInputs.[i]
                calcLayerForward false thetas shape.Activation blockInputs
            ) |> List.unzip)
            let out, net = (lout |> DenseMatrix.append), (lnet |> DenseMatrix.append)
            ForwardResultHidden({ Thetas = thetas; Net = net; Out = out; Shape = shape })

    ) (ForwardResultInput({ Inputs = inputs }))

let forward (inputs: FMatrix) (shape: NNShape) (theta: FVector) =
    reshapeNN shape theta
    |> (forward2 inputs)

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

type BackpropGradient =
    | BackpropGradientEmbed of FMatrix list

type BackpropOutputLayerResult = {
    Thetas : FMatrix
    Delta : FMatrix // delats for each sample in batch
}

type BackpropHiddenLayerResult = {
    Thetas : FMatrix
    Gradient : FMatrix
    Delta : FMatrix // delats for each sample in batch
    Shape : NNLayerShape
}

type BackpropInputLayerResult = {
    Gradient : FMatrix
}

//BackpropResult
type BackpropResult =
    | BackpropResultOutput of BackpropOutputLayerResult
    | BackpropResultHidden of BackpropHiddenLayerResult
    | BackpropResultInput of BackpropInputLayerResult

let private caclGrads useBias (aᴸ: FMatrix) (δᴸᴾ: FMatrix) =
    let aᴸ = iif useBias (aᴸ |> appendOnes) aᴸ
    let δᴸᴾ = δᴸᴾ.Transpose()
    δᴸᴾ * aᴸ

let private caclHiddenDelta fwdLayer (wᴸᴾ: FMatrix) δᴸᴾ =
    let ΔE_ΔA = δᴸᴾ * wᴸᴾ.RemoveColumn(0)
    let ΔA_ΔN = fwdLayer.Net |> mapRows fwdLayer.Shape.Activation.f'
    ΔE_ΔA .* ΔA_ΔN

//TODO : inputs must already contain bias (check FBiasVector)
let private _backprop restrictFn (Y: FMatrix) (X: FMatrix) (shape: NNShape) (theta: FVector) =

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

    let fwdResult = forward2 X layers

    //calc backprop result for the output layer
    let outputLayerBackpropResult =
        match fwdResult |> Array.last with
        | ForwardResultHidden(l) ->
            let ΔE_ΔA = l.Out - Y
            let ΔA_ΔN =  l.Net |> mapRows l.Shape.Activation.f'
            let δᴸ = ΔE_ΔA .* ΔA_ΔN
            BackpropResultOutput({ Thetas = l.Thetas; Delta = δᴸ; })
        | _ -> failwith "Last layer in forard prop must be ForwardHiddenLayerResult"

    Array.scanBack (fun fwd pervBack ->
        match fwd, pervBack with
        | (ForwardResultHidden(l), BackpropResultOutput({Thetas = wᴸᴾ; Delta = δᴸᴾ; }))
        | (ForwardResultHidden(l), BackpropResultHidden ({Thetas = wᴸᴾ; Delta = δᴸᴾ; })) ->
            // current layer is hidden; pervious is output or hidden
            match l.Shape with
            | NNFullLayerShape _ ->
                let δᴸ = caclHiddenDelta l wᴸᴾ δᴸᴾ
                let Δᴸ = caclGrads true l.Out δᴸᴾ
                BackpropResultHidden({ Thetas = l.Thetas; Delta =  δᴸ; Gradient = Δᴸ; Shape = l.Shape})
            | NNEmbedLayerShape _ ->
                failwith "No support for fully connected layer -> embed layer for hidden layers"
        | (ForwardResultInput(l), BackpropResultOutput ({Delta = δᴸᴾ;})) ->
            let Δᴸ = caclGrads true l.Inputs δᴸᴾ
            BackpropResultInput({ Gradient = Δᴸ })
        | (ForwardResultInput(l), BackpropResultHidden({Delta =  δᴸᴾ; Thetas = wᴸᴾ; Shape = shapeᴸᴾ})) ->
            match shapeᴸᴾ with
            | NNFullLayerShape _ ->
                let Δᴸ = caclGrads true l.Inputs δᴸᴾ
                BackpropResultInput({ Gradient = Δᴸ })
            | NNEmbedLayerShape { BlocksNumber = blocksNumber } ->
                // array of matrix sigmas for each emabed layer
                let chunkedDeltas = δᴸᴾ |> chunkColumns2 blocksNumber
                // array of matrix inputs for each emabed layer
                let chunkedInputs = l.Inputs |> chunkColumns2 blocksNumber
                // gradients for each block
                let Δᴸ = Array.map2 (caclGrads false) chunkedInputs chunkedDeltas |> List.ofArray
                BackpropResultInput({ Gradient = BackpropGradientEmbed(Δᴸ) |> restrictFn })
        | _ ->
           failwith "not supported"
    ) fwdResult.[0..fwdResult.Length - 1] outputLayerBackpropResult


let private backpropRestrictGrads grads =
    match grads with
    | BackpropGradientEmbed embedGrads ->
        embedGrads
        |> foldByColumns (fun gradsByEmbedLayer ->
            //calc mean of the related grads in different embed blocks, and set them for each embed layer the same
            // [Δw1; Δw2]  - first embed layer gards
            // [Δw3; Δw4]  - snd embed layer grads
            // then shared gards for Δw1 and Δw3 (since they must be the same) Δw1 = Δw3 = (Δw1 + Δw3) / 2

            // Equal number of embed layers
            let embedBlocksNumber = gradsByEmbedLayer |> Seq.length |> float
            // Equal to number of weights between pervious layer and a single embed layer
            let vecLength = gradsByEmbedLayer |> Seq.item 0 |> Vector.length

            gradsByEmbedLayer
            |> Seq.fold (+) (zeros vecLength)
            |> fun x -> x / embedBlocksNumber
        )



let backprop2 backpropRestrictGrads (y: FVector) (x: FMatrix) (shape: NNShape) (theta: FVector) =
    let Y = chunkOutputs x.RowCount y
    // grads for each sample (per layer)
    _backprop backpropRestrictGrads Y x shape theta
    |> Array.collect(fun res ->
        // avg weighted grads per layer
        match res with
        | BackpropResultHidden { Gradient = gradient }
        | BackpropResultInput { Gradient = gradient } ->
            gradient |> Matrix.map(fun m -> m / float x.RowCount) |> flatMx |> Vector.toArray
        | _ ->
            failwith "not expected"
    )
    |> DenseVector.ofArray

(*
    # Backprop and restrict gardients if necessary
*)
let backprop (y: FVector) (x: FMatrix) (shape: NNShape) (theta: FVector) =
    backprop2 backpropRestrictGrads y x shape theta



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


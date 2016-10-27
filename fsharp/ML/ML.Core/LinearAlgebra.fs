module ML.Core.LinearAlgebra 

open Utils
open Deedle

open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra

type FVector = float Vector

type FMatrix = float Matrix

type NormParams = { Mu : float Vector ; Std : float Vector }

let mx2frame : FMatrix -> Frame<int, int> = Matrix.toArray2 >> Frame.ofArray2D
let frame2mx : Frame<int, int> -> FMatrix =  Frame.toArray2D >> DenseMatrix.ofArray2
let ser2vec : Series<int, float> -> FVector  = Series.values >> DenseVector.ofSeq

let normFrame (frame: Frame<'a, 'b>) =     
    let mu, std = frame |> Stats.mean , frame |> Stats.stdDev
    let nframe = 
        frame 
        |> Frame.mapRowValues (fun r -> (r.As<float>() - mu) / std |> Series.mapAll(fun _ v -> 
            match v with | None -> Some(0.) | _ -> v)) 
        |> Frame.ofRows
    mu, std, nframe

let norm (mx: FMatrix) = 
    let mu, std, frame = mx |> mx2frame |> normFrame   
    frame |> frame2mx, { Mu = mu |> ser2vec; Std = std |> ser2vec}

(*
let norm (mx: float Matrix) : float Matrix * NormParams = 
    
    let mu, std = 
        mx.EnumerateColumns()
        |> Seq.map (fun col -> 
            col.Mean(), col.StandardDeviation()
        )
        |> Seq.toList
        |> List.unzip
                   
    mx |> Matrix.mapCols (fun i vec ->
        if std.[i] <> 0. then
            (vec - mu.[i]) / std.[i]
        else 
            vec - mu.[i]
    ), { Mu = vector mu; Std = vector std }

let norm2 (prms: NormParams) (mx: float Matrix)  : float Matrix = 
    DenseMatrix.initRows mx.RowCount (fun i ->
        (mx.Row(i) - prms.Mu) ./ prms.Std
   )
*)


let spliceRows start count (mx: _ Matrix) = 
    mx.SubMatrix(start, count, 0, mx.ColumnCount)

let spliceVector start count (vr: _ Vector) = 
    vr.SubVector(start, count)

let zeros cnt = DenseVector.zero<float> cnt

let rndvec2 next cnt  = 
    let rnd = new System.Random() 
    DenseVector.init cnt (fun _ -> (next rnd) )

let rndvec cnt = rndvec2 nextGaussianStd cnt

let empty() = DenseVector.create 1 0.
let emptyM() = DenseMatrix.create 1 1 0.

let zerosMatrix (rows, cols) = 
    matrix(List.init rows (fun r -> List.init cols (fun _ -> 0.)))

let flat (mx: _ Matrix) =
    vector(mx.Enumerate() |> Seq.toList)  

(**
Reshape vector into [rows * cols] matrix
**)
let reshape (rows, cols) (vec: _ Vector)  = 
    let mutable cnt = -1
    DenseMatrix.init rows cols (fun i j -> 
        cnt <- cnt + 1
        vec.[cnt]             
    )


let ones cnt = DenseVector.init cnt (fun _ -> 1.)

let appendOnes (mx: _ Matrix) = mx.InsertColumn(0, ones mx.RowCount)

let removeFirstColumn (mx: _ Matrix) = mx.RemoveColumn(0)

let vecCons (el: _) (vec: _ Vector)  = Vector.insert 0 el vec

let appendOne (vec: _ Vector) = vecCons 1. vec     

let repvec (cnt : int) (vec: _ Vector) =
    List.init cnt (fun _ -> vec.Enumerate() |> Seq.toList)
    |> List.concat
    |> vector

let repmatCol (cnt: int) (mx : _ Matrix) =         
    mx.EnumerateRows()
    |> Seq.toList
    |> List.map (fun row -> row |> repvec cnt |> Vector.toList)
    |> matrix

let repmatRow (cnt: int) (mx : _ Matrix) =         
    List.init cnt (fun _ -> mx.EnumerateRows() |> Seq.map(fun f ->  f |> Vector.toList))
    |> Seq.concat 
    |> Seq.toList       
    |> matrix


let oneHot (vec: FVector) = 
    let z = zeros vec.Count
    z.[vec.MaximumIndex()] <- 1.
    z


let encodeOneHot (classesNum: int) (labels: float Vector) = 
    List.init labels.Count (fun r -> 
        List.init classesNum (fun c -> 
            if labels.[r] - 1. = float c then 
                1. 
             else 
                0.
        )
    )
    |> matrix 

// For nn each sample output contains many nodes (number of nodes)
// Outputs still presented as a flat vector, we need to chunk this vector for each sample         

let chunkOutputs samplesNumber y =
    let chunkSize = (y |> Seq.length) / samplesNumber
    y |> reshape (samplesNumber, chunkSize)
        
let permuteSamples (mx: _ Matrix) (vec: _ Vector) =

    //prepare vec for prmute [1; 2; 3; 4] -> [[1;2];[3;4]] and then permute rows
    let perm = new MathNet.Numerics.Permutation (permute mx.RowCount)
    let clonedMx = mx.Clone()        
    let clonedVec = vec |> chunkOutputs mx.RowCount
    clonedMx.PermuteRows(perm)
    clonedVec.PermuteRows(perm)
    //flat permuted matrix output to vector again
    //[[1;2];[3;4]] -> [3; 4; 1; 2]
    let vecRes = clonedVec |> Matrix.toColSeq |> Seq.collect (fun f -> f) |> DenseVector.ofSeq
    clonedMx, vecRes

let flatMx (mx: FMatrix) =
    mx.EnumerateColumns() |> Seq.concat |> DenseVector.ofSeq

let flatMxs (mxs: FMatrix array) =
    mxs |> Seq.collect flatMx |> DenseVector.ofSeq

let mapRows f (mx: FMatrix) =
    mx |> Matrix.mapRows (fun i r -> f r)

let normalizeMx (mx: FMatrix) = 
    mx.NormalizeColumns(2.)

let chunkColumns chunkSize (mx : FMatrix) =
    mx |> Matrix.toColArrays |> Array.chunkBySize chunkSize |> Array.map DenseMatrix.ofColumnSeq

let chunkColumns2 chunksNumber (mx : FMatrix) =
    chunkColumns (mx.ColumnCount / chunksNumber) mx
    
let appendColumns (mx1 : FMatrix) (mx2 : FMatrix) =
    DenseMatrix.append([mx1; mx2])

let foldByColumns f (mxs: FMatrix seq) =
    mxs 
    |> Seq.item 0
    |> Matrix.mapCols(fun i _ -> 
      mxs 
      |> Seq.map(fun mx -> mx.Column(i))  
      |> f
    ) 
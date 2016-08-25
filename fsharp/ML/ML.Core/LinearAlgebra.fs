module ML.Core.LinearAlgebra 

open Utils

open MathNet.Numerics.LinearAlgebra

type FVector = float Vector

type FMatrix = float Matrix

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

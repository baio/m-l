module ML.Core.LinearAlgebra 

    open Utils

    open MathNet.Numerics.LinearAlgebra

    let spliceRows start count (mx: _ Matrix) = 
        mx.SubMatrix(start, count, 0, mx.ColumnCount)

    let spliceVector start count (vr: _ Vector) = 
        vr.SubVector(start, count)

    let zeros cnt = DenseVector.zero<float> cnt

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


    let encodeOneHot (classesNum: int) (labels: float Vector) = 
        List.init labels.Count (fun r -> 
            List.init classesNum (fun c -> 
                if labels.[r] = float c then 1. else 0.
            )
        )
        |> matrix 
        
    let permuteSamples (mx: _ Matrix) (vec: _ Vector) =
        let perm = new MathNet.Numerics.Permutation (permute mx.RowCount)
        let cloned = mx.InsertColumn(0, vec)        
        cloned.PermuteRows(perm)
        cloned.RemoveColumn(0), cloned.Column(0)
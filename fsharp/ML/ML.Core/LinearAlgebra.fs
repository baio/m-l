module ML.Core.LinearAlgebra 

    open MathNet.Numerics.LinearAlgebra

    let spliceRows start count (mx: _ Matrix) = 
        mx.SubMatrix(start, count, 0, mx.ColumnCount)

    let spliceVector start count (vr: _ Vector) = 
        vr.SubVector(start, count)

    let zeros cnt = DenseVector.zero cnt

    let zerosMatrix (rows, cols) = 
        matrix(List.init rows (fun r -> List.init cols (fun _ -> 0.)))

    let flat (mx: _ Matrix) =
        vector(mx.Enumerate() |> Seq.toList)  

    let reshape (rows, cols) (vec: _ Vector)  = 
        let mutable cnt = -1
        matrix(List.init rows (fun r -> 
            List.init cols (fun c -> 
                    cnt <- cnt + 1
                    vec.[cnt]                    
                )
            )
        )

    let ones cnt = DenseVector.init cnt (fun _ -> 1.)

    let appendOnes (mx: _ Matrix) = mx.InsertColumn(0, ones mx.RowCount)

    let vecCons (el: _) (vec: _ Vector)  = Vector.insert 0 el vec

    let appendOne (vec: _ Vector) = vecCons 1. vec     

    let repmatCol (cnt: int) (mx : _ Matrix) = 
        Seq.init cnt (fun f -> mx.EnumerateColumns())
        |> Seq.concat
        |> Seq.toList 
        |> List.map (fun v -> v.ToArray() |> List.ofArray)
        |> matrix

    let encodeOneHot (classesNum: int) (labels: float Vector) = 
        List.init labels.Count (fun r -> 
            List.init classesNum (fun c -> 
                if labels.[r] = float c then 1. else 0.
            )
        )
        |> matrix 
        
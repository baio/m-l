module ML.Core.LinearAlgebra 

    open MathNet.Numerics.LinearAlgebra

    let spliceRows start count (mx: _ Matrix) = 
        mx.SubMatrix(start, count, 0, mx.ColumnCount)

    let spliceVector start count (vr: _ Vector) = 
        vr.SubVector(start, count)


    let zeros cnt = DenseVector.zero cnt

    let ones cnt = DenseVector.init cnt (fun _ -> 1.)

    let appendOnes (mx: _ Matrix) = mx.InsertColumn(0, ones mx.RowCount)

    let vecCons (el: _) (vec: _ Vector)  = Vector.insert 0 el vec

    let appendOne (vec: _ Vector) = vecCons 1. vec 
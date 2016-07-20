module ML.Core.Utils

open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra

type NormParams = { Mu : float Vector ; Std : float Vector }

let norm (mx: float Matrix) : float Matrix * NormParams = 
    
    let mu, std = 
        mx.EnumerateColumns()
        |> Seq.map (fun col -> 
            col.Mean(), col.StandardDeviation()
        )
        |> Seq.toList
        |> List.unzip
                   
    mx |> Matrix.mapCols (fun i vec ->
        (vec - mu.[i]) / std.[i]
    ), { Mu = vector mu; Std = vector std }



let genRanges rgLength seqLength =
    seq {         
        for i in 0..rgLength..seqLength do 
            if i <> seqLength then
                yield i, if i + rgLength <= seqLength then rgLength else seqLength - i
    }

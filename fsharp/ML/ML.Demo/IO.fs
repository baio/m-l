module IO

open ML.Core.LinearAlgebra
open ML.Core.Readers
open ML.Core.Utils
open MathNet.Numerics.LinearAlgebra
open Deedle

let getCols cols pred = 
    cols
    |> Seq.mapi (fun i m -> i, m)
    |> Seq.filter(fun (i, _) -> pred i)
    |> Seq.map snd


let normalizeCsv inFile outFile ignoreCols = 
    
    let isIgnoreCol i = ignoreCols |> (List.contains i >> not)
    let data = Frame.ReadCsv(path = inFile, hasHeaders = false)    
    let getDataCols = getCols data.ColumnKeys
    let normCols = getDataCols isIgnoreCol
    let notNormCols = getDataCols (isIgnoreCol >> not)
   
    let mu, std, normData = data.Columns.[normCols] |> normFrame
    let notNormData = data.Columns.[notNormCols] |> Frame.mapCols(fun _ s -> s.As<float>())

    let resFrame = notNormData + normData
    resFrame.SaveCsv(path = outFile)



module ML.Utils
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra

let readLines (filePath:string) = seq {
    use sr = new System.IO.StreamReader (filePath)
    while not sr.EndOfStream do
        yield sr.ReadLine ()
}

// raed csv, all columns contain decimal numbers
let readCSV (filePath:string) isHeader (inputCols: int []) (outputCol: int)= 
    
    let mapLine (str: string) = 
        let cols = str.Split([|','|])
        let outs = seq { for i in inputCols -> System.Double.Parse(string cols.[i]) } |> Seq.toList
        outs, System.Double.Parse(cols.[outputCol])
        
    readLines filePath
    |> Seq.skip (if isHeader then 1 else 0)
    |> Seq.map mapLine
    |> Seq.toList
    |> List.unzip        


let norm (mx: float Matrix) = 
    
    let colsStat = 
        mx.EnumerateColumns()
        |> Seq.map (fun col -> 
            col.Mean(), col.StandardDeviation()
        )
        |> Seq.toArray
    
    mx    
    |> Matrix.mapCols (fun i vec ->
        let mean, std = colsStat.[i]
        (vec - mean) / std
    ), colsStat
        

let zeros cnt = DenseVector.zero cnt

let ones cnt = DenseVector.init cnt (fun _ -> 1.)

let appendOnes (mx: _ Matrix) = mx.InsertColumn(0, ones mx.RowCount)
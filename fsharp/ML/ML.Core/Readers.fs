module ML.Core.Readers

open Utils
open Nessos.Streams

let readLines (filePath:string) = seq {
    use sr = new System.IO.StreamReader (filePath)
    while not sr.EndOfStream do
        yield sr.ReadLine ()
}

// raed csv, all columns contain decimal numbers
let readCSV (filePath:string) isHeader (inputCols: int []) (outputCol: int) = 
    
    let mapLine (str: string) = 
        let cols = str.Split([|','|])
        let outs = seq { for i in inputCols -> System.Double.Parse(string cols.[i]) } |> Seq.toList
        outs, System.Double.Parse(cols.[outputCol])
        
    readLines filePath
    |> Seq.skip (if isHeader then 1 else 0)
    |> Seq.map mapLine
    |> Seq.toList
    |> List.unzip        


let readCSV2 (filePath:string) isHeader (inputCols: int []) (outputCol: int) (cnt: int) = 
    
    let mapLine (str: string) = 
        let cols = str.Split([|','|])
        let outs = seq { for i in inputCols -> System.Double.Parse(string cols.[i]) } |> Seq.toList
        outs, System.Double.Parse(cols.[outputCol])
        
    readLines filePath
    |> Seq.skip (if isHeader then 1 else 0)
    |> Seq.take cnt
    |> Seq.map mapLine
    |> Seq.toList
    |> List.unzip        

let readCSV3 (filePath:string) isHeader (inputCols: int []) (outputCol: int) (outputCats: Map<string, float>) = 
    
    let mapLine (str: string) = 
        let cols = str.Split([|','|])
        let outs = seq { for i in inputCols -> System.Double.Parse(string cols.[i]) } |> Seq.toList
        outs, outputCats.[cols.[outputCol]]
        
    readLines filePath
    |> Seq.skip (if isHeader then 1 else 0)
    |> Seq.map mapLine
    |> Seq.toList
    |> List.unzip        


let foldStream colsNumber cernel (stream : float list Stream) = 
    stream 
    |> Stream.fold 
        (fun acc v -> acc |> List.mapi (fun i e -> cernel e v.[i])) 
        (List.init colsNumber (fun _ -> 0.))

let sumStream colsNumber = foldStream colsNumber (fun a b -> a + b) 

let mapStream cernel (stream : float list Stream) = 
    stream 
    |> Stream.map
        (fun m -> m |> List.mapi (fun i e -> cernel i e)) 
    
let meanList rowsNumber lst = lst |> List.map (fun m -> m / rowsNumber)

let meanSumStream rowsNumber colsNumber stream = 
    stream |> sumStream colsNumber |> meanList rowsNumber

//given stream of columns return mean and std dev for each column
let stdDevStream rowsNumber colsNumber (stream : float list Stream) =       
    let mss = meanSumStream rowsNumber colsNumber
    //Work out the Mean (the simple average of the numbers)
    let mu = stream |> mss
    //Then for each number: subtract the Mean and square the result.    
    let sq = stream |> mapStream (fun i e -> System.Math.Pow(e - mu.[i], 2.))  
    // Then work out the mean of those squared differences.    
    let sqsum = sq |> mss
    //Take the square root of that and we are done!
    mu, sqsum |> List.map (fun m -> System.Math.Sqrt m)

//given stream of columns return normalized columns plus mean and std dev for each column (normalized, (mu, std))
let normStream rowsNumber colsNumber (stream : float list Stream) =       
    let mu, std = stream |> stdDevStream rowsNumber colsNumber
    let nrm = stream |> mapStream (fun i e -> (e - mu.[i]) / std.[i])
    nrm, (mu, std)
      

//The purpose of this to not optimize normalization of the columns 
//but make it memory efficient, since this func should work with big files
//so we need proccess file line by line
//It is slow but, consume least memory
//TODO : test
//TODO : write first line with comment #RowsNum, NormMu, NormStdDev
let normalizeFile rowsNumber normIndexes unnormIndex pathIn pathOut = 
    
    let takeByIndexes (s : _ seq) =
        List.init (normIndexes |> List.length)  (fun i -> s |> Seq.item normIndexes.[i])

    use sw = new System.IO.StreamWriter (path = pathOut)

    let rows = 
        readLines pathIn
        |> Stream.ofSeq
        |> Stream.map (fun m ->
            let spts = m.Split(',') 
            spts.[unnormIndex], spts|> takeByIndexes |> List.map (fun m -> System.Double.Parse(m)) 
        )

    let y = rows |> Stream.map (fun m -> fst m)
    let x, norm = rows |> Stream.map (fun m -> snd m) |> normStream rowsNumber normIndexes.Length

    x 
    |> Stream.zipWith (fun a b -> a, b) y
    |> Stream.iter (fun (y, x) ->
        y::(x |> List.map (fun f -> sprintf "%f" f))
        |> String.concat ";"
        |> sw.WriteLine
    )

    //TODO : write meta file

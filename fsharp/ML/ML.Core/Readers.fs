module ML.Core.Readers

//#I "../../packages/Deedle"
//


open Utils
open ML.Core.LinearAlgebra
open Nessos.Streams

let readLines (filePath:string) = seq {
    use sr = new System.IO.StreamReader (filePath)
    while not sr.EndOfStream do
        yield sr.ReadLine ()
}

let inline parseEntry entry =
    if entry = "" then 0. else System.Double.Parse(string entry)

//TODO: rename readCSV4
let readCSVFloats (filePath:string) isHeader =

    readLines filePath
    |> Seq.skip (if isHeader then 1 else 0)
    |> Seq.map (split2 "," >> Seq.map(System.Double.Parse))
    

// raed csv, all columns contain decimal numbers
let readCSV (filePath:string) isHeader (inputCols: int []) (outputCol: int) =

    let mapLine (str: string) =
        let cols = str |> split2 ","
        let outs = seq { for i in inputCols -> parseEntry cols.[i] } |> Seq.toList
        outs, System.Double.Parse(cols.[outputCol])

    readLines filePath
    |> Seq.skip (if isHeader then 1 else 0)
    |> Seq.map mapLine
    |> Seq.toList
    |> List.unzip


let readCSV2 (filePath:string) isHeader (inputCols: int []) (outputCol: int) (cnt: int) =

    let mapLine (str: string) =
        let cols = str.Split([|','|])
        let outs = seq { for i in inputCols -> parseEntry cols.[i] } |> Seq.toList
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

(*
let meanList (rowsNumber: int) lst = lst |> List.map (fun m -> m / float rowsNumber)

let meanSumStream rowsNumber colsNumber stream =
    stream |> sumStream colsNumber |> meanList rowsNumber

//given stream of columns return mean and std dev for each column
let stdDevStream (rowsNumber: int) colsNumber (stream : float list Stream) =
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
let normStream (rowsNumber: int) colsNumber (stream : float list Stream) =
    let mu, std = stream |> stdDevStream rowsNumber colsNumber
    let nrm = stream |> mapStream (fun i e -> (e - mu.[i]) / std.[i])
    nrm, (mu, std)


//The purpose of this to not optimize normalization of the columns
//but make it memory efficient, since this func should work with big files
//so we need proccess file line by line
//It is slow but, consume least memory
//TODO : test
//TODO : write first line with comment #RowsNum, NormMu, NormStdDev

let normalizeStream (rowsNumber: int) normIndexes unnormIndex (streamIn : string Stream) =

    let takeByIndexes (s : _ seq) =
        List.init (normIndexes |> List.length)  (fun i -> s |> Seq.item normIndexes.[i])

    let rows =
        streamIn
        |> Stream.map (fun m ->
            let spts = m.Split(',')
            let nromzd =
                spts
                |> takeByIndexes
                |> List.map System.Double.Parse
            spts.[unnormIndex], nromzd
        )

    let y = rows |> Stream.map fst
    let x, norm = rows |> Stream.map snd |> normStream rowsNumber normIndexes.Length

    x
    |> Stream.zipWith (fun a b -> a, b) y
    |> Stream.map (fun (y, x) ->
        y::(x |> List.map (fun f -> sprintf "%f" f)) |> String.concat ","
    )
*)


(*
let normalizeFile rowsNumber normIndexes unnormIndex pathIn pathOut =

    let takeByIndexes (s : _ seq) =
        List.init (normIndexes |> List.length)  (fun i -> s |> Seq.item normIndexes.[i])

    let mapLine line =
        line.Split(',')
        |>


    //use sw = new System.IO.StreamWriter (path = pathOut)

    readLines pathIn
    |> Stream.ofSeq
    |> normalizeStream rowsNumber normIndexes unnormIndex
    |> Stream.iter sw.WriteLine
*)



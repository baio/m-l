module WordEmbed
open ML.Core.Readers
open ML.Core.Utils
open ML.Core.LinearAlgebra
open MathNet.Numerics.LinearAlgebra

let writeSparseCSV inputFileName outputFileName =
    let lines = 
        inputFileName
        |> readLines 
        |> Seq.map (toLower >> split >> Array.take 4 >> Seq.ofArray)

    let words = lines |> Seq.concat |> Set.ofSeq |> Seq.mapi tulpeFlip |> Map.ofSeq
    let wordsCount = words.Count

    use file = new System.IO.StreamWriter(outputFileName, false);
    lines 
    |> Seq.map(Seq.map(fun w -> words.[w]))
    |> Seq.iter (fun row ->
        row 
        |> Seq.map(sprintf "%i") 
        |> String.concat "," 
        |> file.WriteLine
    )    

let wordEmded () = 
    //uncoment to gen
    //writeSparseCSV "../../word_embed_sentences.txt" "c:/dev/.data/word_embed_sentences.csv"

    let x, y =
        readCSVFloats @"c:/dev/.data/word_embed_sentences.csv" false
        |> DenseMatrix.ofRowSeq
        |> splitCols 3

    let xs = 
        x |> Matrix.scanRows(fun _ v ->            
            v |> Seq.collect(int >> sparse 250) |> Seq.toList
        ) []
        |> Seq.skip(1)
        |> Seq.toList
    
    ignore

    
    


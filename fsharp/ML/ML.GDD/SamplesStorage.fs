module ML.DGD.SamplesStorage
open ML.Core.Utils
open ML.Core.LinearAlgebra
open MathNet.Numerics.LinearAlgebra

type SamplesStorageCloud =
    | SamplesStorageCloudAzure

//Path and meta of samples storage
//Comma separated path to local or cloud blobs
type SamplesStorageLocation =
    | SamplesStorageFile of string
    | SamplesStorageCloud of SamplesStorageCloud

type SamplesStorage = {
    Location: SamplesStorageLocation
    //Number of samples
    //SamplesNumber: int
    //Feature columns
    Features : int list
    //Label column
    Label : int
}

let readSamplesFile (path: string) (indexes: int list option) : string seq =    
    let mutable i = 0
    seq {
        use sr = new System.IO.StreamReader (path)
        while not sr.EndOfStream do
            let l = sr.ReadLine()
            match indexes with
            | Some ixs ->
                if ixs |> List.exists (fun f -> f = i) then
                    yield l
            | None ->
                yield l
    } |> Seq.take 5000

let takeByIndexes (indexes: int list) (s : _ seq) =
    seq {for i in indexes -> (s |> Seq.nth i)}            

let readSamples (storage: SamplesStorage) (indexes: int list option) =
    printfn "read"
    let x, y = 
        match storage.Location with
        | SamplesStorageFile path -> 
            readSamplesFile path indexes 
        | _ -> failwith "not implemented"    
        |> Seq.map (fun m ->
            let x = 
                m.Split(',') 
                |> takeByIndexes (storage.Label::storage.Features) 
                |> Seq.map (fun m -> System.Double.Parse(m)) 
            (x |> Seq.skip 1 |> List.ofSeq), (x |> Seq.head)
        )
        |> List.ofSeq
        |> List.unzip

    let x, nrmPrms = x |> DenseMatrix.ofRowList |> norm
    let y = y |> DenseVector.ofList
    permuteSamples x y

module IO

open ML.Core.Readers
open ML.Core.Utils

open Deedle

let IO() = 
    let iris = Frame.ReadCsv("c:/dev/.data/mnist/mnist_train.csv")
    //let iris = Frame.ReadCsv("c:/dev/.data/iris.csv")
    let keys = iris.ColumnKeys |> Seq.toArray
    let x = iris.Columns.[keys.[1..784]]

    let iris = None

    let mu = x |> Stats.mean 
    let std = x |> Stats.stdDev
       
    let norm = 
      x 
      |> Frame.mapRowValues (fun r -> (r.As<float>() - mu) / std)
      |> Frame.ofRows

    let x = None

    printfn "%A" norm


#I "..\packages\Deedle.1.2.5"
#I "..\packages\FSharp.Charting.0.90.14"
#I "..\packages\MathNet.Numerics.3.13.1"
#I "..\packages\MathNet.Numerics.3.13.1"
#I "..\packages\MathNet.Numerics.FSharp.3.13.1"

#load "MathNet.Numerics.fsx"
#load "FSharp.Charting.fsx"
#load "Deedle.fsx"

open System
open Deedle
open FSharp.Charting
open MathNet.Numerics.LinearAlgebra

type Rec = {x: int; y: int}

type FMatrix = Matrix<float>
type FVector = Vector<float>

let x = seq {
    for i in [0..10] -> {x  = i; y = i * i}
}

let f = Frame.ofRecords x

let mu = f |> Stats.mean
let std = f |> Stats.stdDev

let calcP s = s |> Seq.mapi (fun i x -> mu |> Series.getAt i)

let findAnomaly (mx: FMatrix) =>
  mx.

module ML.Core.Norm.Tests

open Xunit
open FsUnit
open ML.Core.Statistics

type Batch = float seq seq 

let batchOfList = Seq.ofList >> Seq.map Seq.ofList

let calcBtachNorm (calcNorm: Distr -> NormModel) (batch: Batch) =    
    batch 
    |> Seq.fold (fun acc v ->
        v |> Seq.map distr |> Seq.zip acc |> Seq.map (fun (a, b) -> a ++ b)
    ) (Seq.initInfinite (fun _ -> distrZero()))

let calcBtachNormModel (calcNorm: Distr -> NormModel) (batch: Batch) =    
    batch |> calcBtachNorm calcNorm |> Seq.map calcNorm

let calcBtachNormModelP : Batch -> NormModel seq = calcBtachNormModel calcNormModelP 

let calcBtachNormModelS : Batch -> NormModel seq = calcBtachNormModel calcNormModelS
       
[<Fact>]
let ``Calc norm p model ``() =

    let actual = [ [1.;2.]; [3.;4.] ] |> batchOfList |> calcBtachNormModelP

    let expected = [ { Mu = 2. ; Std = 1. }; { Mu = 3. ; Std = 1. } ]

    actual |> should equal expected


[<Fact>]
let ``Calc norm s model ``() =

    let actual = [ [1.;2.]; [3.;4.] ] |> batchOfList |> calcBtachNormModelS

    let expected = [ { Mu = 2. ; Std = 1.4142135623730951 }; { Mu = 3. ; Std = 1.4142135623730951 } ]

    let x = actual |> List.ofSeq

    actual |> should equal expected
    


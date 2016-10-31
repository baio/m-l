module ML.Core.Norm.Tests

open NUnit.Framework
open FsUnit
open ML.Core.Statistics
open ML.Core.Normalization
open ML.Core.Batch.Statistics


[<TestCase>]
let ``Calc norm p model ``() =

    let actual = [ [1.;2.]; [3.;4.] ] |> batchOfList |> calcBtachNormModelP

    let expected = [ { Mu = 2. ; Std = 1. }; { Mu = 3. ; Std = 1. } ]

    actual |> should equal expected


[<TestCase>]
let ``Calc norm s model ``() =

    let actual = [ [1.;2.]; [3.;4.] ] |> batchOfList |> calcBtachNormModelS

    let expected = [ { Mu = 2. ; Std = 1.4142135623730951 }; { Mu = 3. ; Std = 1.4142135623730951 } ]

    let x = actual |> List.ofSeq

    actual |> should equal expected
    

[<TestCase>]
let ``Normalize p model ``() =

    let actual = [ [1.;2.]; [3.;4.] ] |> batchOfList |> normalizeBatchP 

    let expected = [ [-1.; -1.]; [1.; 1.] ]

    let x = actual |> List.ofSeq

    actual |> should equal expected
    


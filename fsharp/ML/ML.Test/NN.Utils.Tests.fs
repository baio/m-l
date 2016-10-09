module ML.NN.Utils.Test

open Xunit
open FsUnit
open MathNet.Numerics.LinearAlgebra
open NUnit.Framework.Constraints
open ML.Core.Utils
open ML.Core.LinearAlgebra

(*
type Range = Within of float * float
let (+/-) (a:float) b = Within(a, b)

let equal x = 
  match box x with 
  | :? Range as r ->
      let (Within(x, within)) = r
      (new EqualConstraint(x)).Within(within)
  | _ ->
    new EqualConstraint(x)
*)

let round i (v: float) = System.Math.Round(d = decimal v, decimals = i) |> float
let rounds i = Seq.map (round i)
let vround i = Vector.toSeq >> rounds i >> DenseVector.ofSeq
let vr8 () = vround 8
let vr9 () = vround 9

[<Fact>]
let ``Chunk outputs must work``() =

    let mx = matrix [[0.; 1.]; [1.; 0.]; [7.; 3.;]]
    let actual = mx |> flatMx |> chunkOutputs mx.RowCount

    actual |> should equal mx


[<Fact>]
let ``Norm should work``() =

    let mx = [[1.; 2.]; [3.; 4.]] |> DenseMatrix.ofRowList
    let actual, _ = mx |> norm
    let expected = [[-0.70710678118654746; -0.70710678118654746]; [0.70710678118654746; 0.70710678118654746]] |> DenseMatrix.ofRowList

    actual |> should equal expected


[<Fact>]
let ``Norm with 0 std should work``() =
    let mx = [[1.; 2.]; [1.; 2.]] |> DenseMatrix.ofRowList
    let actual, prms = mx |> norm
    let expected = [[0.; 0.]; [0.; 0.]] |> DenseMatrix.ofRowList
    let expectedPrms = { Mu = [1.; 2.] |> vector ; Std = [0.; 0.] |> vector}
    prms |> should equal expectedPrms
    actual |> should equal expected

[<Fact>]
let ``Normalize list of rows``() =
    let x = [
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;38 ;222;225
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
        132;255;225;12
        0  ;0  ;0  ;0
        0  ;0  ;0  ;0
    ]

    let nzd, prms =
        x
        |> List.map float
        |> List.chunkBySize 4
        |> DenseMatrix.ofRowList
        |> norm

    let expectedPrms = { Mu = [ 3.77142857; 8.37142857; 12.77142857; 6.77142857 ] |> vector; Std = [22.31207232; 43.39157259; 52.63642539; 38.02637298] |> vector }
    
    let ractual = { Mu = prms.Mu |> vr8(); Std = prms.Std |> vr8() } 

    ractual |> should equal expectedPrms




module Core.IO.Tests

open Xunit
open FsUnit

open MathNet.Numerics.LinearAlgebra
open Nessos.Streams
open ML.Core.Readers
open ML.Core.LinearAlgebra

//[<Fact>]
let ``Normalize stream must workd``() =
    
        
    let actual = matrix [[1.; 1.]; [4.; 2.]] |> normalizeMx 
    let expected = matrix [[0.; 0.]; [0.; 0.]] 
    
    actual |> should equal expected


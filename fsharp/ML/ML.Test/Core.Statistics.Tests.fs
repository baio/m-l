module ML.Core.Statistics.Tests

open Xunit
open FsUnit
open MathNet.Numerics.LinearAlgebra
open NUnit.Framework.Constraints
open ML.Core.Statistics
open ML.Core.LinearAlgebra


[<Fact>]
let ``Calc avg 1. 3. should work``() =

    let d = (distr 1.) ++ (distr 3.)

    calcAvg d |> should equal 2.

[<Fact>]
let ``Calc avg 2. 5. should work``() =

    let d = (distr 2.) ++ (distr 5.)

    calcAvg d |> should equal 3.5

[<Fact>]
let ``Calc avg 0. 0. should work``() =

    let d = (distr 0.) ++ (distr 0.)

    calcAvg d |> should equal 0.

[<Fact>]
let ``Calc var p 1. 3. should work``() =

    let d = (distr 1.) ++ (distr 3.)

    let actual = calcVarP d 
    
    actual |> should equal 1.

[<Fact>]
let ``Calc var s 1. 3. should work``() =

    let d = (distr 1.) ++ (distr 3.)

    let actual = calcVarS d 
    
    actual |> should equal 2.

[<Fact>]
let ``Calc std dev p 1. 3. should work``() =

    let d = (distr 1.) ++ (distr 3.)

    let actual = calcStdDevP d 
    
    actual |> should equal 1.

[<Fact>]
let ``Calc std dev s 1. 3. should work``() =

    let d = (distr 1.) ++ (distr 3.)

    let actual = calcStdDevS d 
    
    actual |> should equal 1.4142135623730951

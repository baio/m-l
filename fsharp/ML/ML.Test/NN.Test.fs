module ML.Test.NN

open Xunit
//open NUnit.Framework
open FsUnit
open MathNet.Numerics.LinearAlgebra

open ML.NN.Types
open ML.NN.NN

let act a = a

[<Fact>]
let ``Reshape for 2 inputs -> 1 output must work``() =
    
    let theta = vector([0.; 0.; 0.])
    
    let shape = {
        Layers = [ { NodesNumber = 2; Activation = act }; { NodesNumber = 1; Activation = act }; ]
    }
    

    let actual = reshapeNN shape theta
    let expected = [| (matrix([ [0.; 0.; 0.] ]), act) |]
    
    Assert.AreEqual(1, 1)
    //actual |> should equal expected



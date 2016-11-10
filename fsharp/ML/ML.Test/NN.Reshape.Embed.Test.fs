module ML.NN.Reshape.Embed

open NUnit.Framework
open FsUnit
open ML.NN
open MathNet.Numerics.LinearAlgebra

open ML.Core.LinearAlgebra

let f a = a
let act = {f = f; f' = f}

[<TestCase>]
let ``Create embed hidden layer [2] -> [1]``() =


    let theta = vector([0.; 1.;])

    let layer =
        NNEmbedLayerShape(
            {
                BlocksNumber = 1
                NodesInBlockNumber = 1
                Activation = act
            }
        )

    let actual = makeHidden theta 2 layer

    actual.Thetas |> should equal (matrix [[0.; 1.]])

[<TestCase>]
let ``Create embed hidden layer [2] -> [2]``() =

    let theta = vector([0.; 1.; 2.; 3.])

    let layer =
        NNEmbedLayerShape(
            {
                BlocksNumber = 1
                NodesInBlockNumber = 2
                Activation = act
            }
        )

    let actual = makeHidden theta 2 layer    

    actual.Thetas |> should equal (matrix([[0.; 2.];[1.;3.]]))

[<TestCase>]
let ``Create embed hidden layer [4] -> [1, 1]``() =

    let theta = vector([0.; 1.])

    let layer =
        NNEmbedLayerShape(
            {
                BlocksNumber = 2
                NodesInBlockNumber = 1
                Activation = act
            }
        )

    let actual = makeHidden theta 4 layer    

    actual.Thetas |> should equal (matrix([[0.; 1.]]))


[<TestCase>]
let ``Create embed hidden layer [0; 1; 2;] -> mx[0; 1; 2]``() =

    let theta = vector([0.; 1.; 2.])

    let layer =
        NNEmbedLayerShape(
            {
                BlocksNumber = 3
                NodesInBlockNumber = 1
                Activation = act
            }
        )

    let actual = makeHidden theta 6 layer

    let expected = matrix([[0.; 1.]]); 

    actual.Thetas |> should equal expected


[<TestCase>]
let ``Create embed hidden layer [4] -> [2, 2]``() =

    let theta = vector([0.; 2.; 1.; 3.;])

    let layer =
        NNEmbedLayerShape(
            {
                BlocksNumber = 2
                NodesInBlockNumber = 2
                Activation = act
            }
        )

    let actual = makeHidden theta 4 layer

    let expectedThetas = matrix([[0.; 1.;];[2.; 3.]]);

    actual.Thetas |> should equal expectedThetas

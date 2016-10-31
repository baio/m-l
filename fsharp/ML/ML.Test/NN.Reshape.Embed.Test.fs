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

    let reshapedLayer = NNLayerReshapeOutput(
        {
            Thetas = [ DenseMatrix.ofRowList([[0.; 1.]]) ]
            Activation= act
        }
     )


    let (NNLayerReshapeOutput ({Thetas = actualThetas})) = makeHidden theta 2 layer


    actualThetas |> should equal [ DenseMatrix.ofRowList([[0.; 1.]]) ]

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

    let (NNLayerReshapeOutput({Thetas = actualThetas})) = actual

    actualThetas |> should equal [ matrix([[0.; 2.];[ 1.; 3.]]) ]

[<TestCase>]
let ``Create embed hidden layer [4] -> [1, 1]``() =

    let theta = vector([0.; 1.; 2.; 3.])

    let layer =
        NNEmbedLayerShape(
            {
                BlocksNumber = 2
                NodesInBlockNumber = 1
                Activation = act
            }
        )

    let actual = makeHidden theta 4 layer

    let (NNLayerReshapeOutput({Thetas = actualThetas})) = actual

    actualThetas |> should equal [ matrix([[0.; 1.]]); matrix([[ 2.; 3.]]) ]


[<TestCase>]
let ``Create embed hidden layer [6] -> [1, 1, 1]``() =

    let theta = vector([0.; 1.; 2.; 3.; 4.; 5.;])

    let layer =
        NNEmbedLayerShape(
            {
                BlocksNumber = 3
                NodesInBlockNumber = 1
                Activation = act
            }
        )

    let actual = makeHidden theta 6 layer

    let (NNLayerReshapeOutput({Thetas = actualThetas})) = actual
    let expectedThetas = [ matrix([[0.; 1.]]); matrix([[ 2.; 3.]]); matrix([[ 4.; 5.]]) ]

    System.Diagnostics.Debug.WriteLine(sprintf "%A" actualThetas)
    System.Diagnostics.Debug.WriteLine(sprintf "%A" expectedThetas)

    actualThetas |> should equal expectedThetas


[<TestCase>]
let ``Create embed hidden layer [4] -> [2, 2]``() =

    let theta = vector([0.; 2.; 1.; 3.; 4.; 6.; 5.; 7.;])

    let layer =
        NNEmbedLayerShape(
            {
                BlocksNumber = 2
                NodesInBlockNumber = 2
                Activation = act
            }
        )

    let actual = makeHidden theta 4 layer

    let (NNLayerReshapeOutput({Thetas = actualThetas})) = actual
    let expectedThetas = [ matrix([[0.; 1.;];[2.; 3.]]); matrix([[ 4.; 5.;];[6.; 7.]]); ]

    System.Diagnostics.Debug.WriteLine(sprintf "%A" actualThetas)
    System.Diagnostics.Debug.WriteLine(sprintf "%A" expectedThetas)

    actualThetas |> should equal expectedThetas

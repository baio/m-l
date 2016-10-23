module ML.NN.ReshapeNN

open Xunit
open FsUnit
open ML.NN
open MathNet.Numerics.LinearAlgebra

open ML.Core.LinearAlgebra

let f a = a
let act = {f = f; f' = f}

[<Fact>]
let ``Create embed hidden layer [2] -> [1]``() =
    
    let theta = vector([0.; 1.;])
    
    let layer = NNEmbedLayerShape({
        BlocksNumber = 1
        NodesInBlockNumber = 1
        Activation = act
    })

    let reshapedLayer = NNLayerReshapeOutput({
            Thetas = [ DenseMatrix.ofRowList([[0.; 1.]]) ]
            Activation= act
        }
     )
    
    
    let (NNLayerReshapeOutput ({Thetas = actualThetas})) = makeHidden theta 2 layer
    
    
    actualThetas |> should equal [ DenseMatrix.ofRowList([[0.; 1.]]) ]

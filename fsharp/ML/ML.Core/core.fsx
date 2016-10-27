//#r "..\packages\MathNet.Numerics.3.13.0\lib\net40\MathNet.Numerics.dll"
#I "..\packages"
#r @"MathNet.Numerics.3.13.1\lib\net40\MathNet.Numerics.dll"
#r @"MathNet.Numerics.FSharp.3.13.1\lib\net40\MathNet.Numerics.FSharp.dll"
#load @"Deedle.1.2.5\Deedle.fsx"
#load "Utils.fs"
#load "LinearAlgebra.fs"


open ML.Core.LinearAlgebra 
open MathNet.Numerics.LinearAlgebra

let mx1 = matrix [[1.; 2.]]
let mx2 = matrix [[3.; 4.]]
let mxs = [mx1; mx2]

let sumvecs vecs = 
    let vecsLength = vecs |> Seq.length |> float
    let vecLength = vecs |> Seq.item 0 |> Vector.length
    vecs
    |> Seq.fold (+) (zeros vecLength)
    |> vecsLength (/)


mxs 
|> Seq.map(fun mx -> mx.Column(0))  
|> sumvecs
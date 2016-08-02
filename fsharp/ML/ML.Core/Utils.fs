module ML.Core.Utils

open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra
open System

type NormParams = { Mu : float Vector ; Std : float Vector }

//normailze with (a - mean / std)
let norm (mx: float Matrix) : float Matrix * NormParams = 
    
    let mu, std = 
        mx.EnumerateColumns()
        |> Seq.map (fun col -> 
            col.Mean(), col.StandardDeviation()
        )
        |> Seq.toList
        |> List.unzip
                   
    mx |> Matrix.mapCols (fun i vec ->
        if std.[i] <> 0. then
            (vec - mu.[i]) / std.[i]
        else 
            vec - mu.[i]
    ), { Mu = vector mu; Std = vector std }

let norm2 (prms: NormParams) (mx: float Matrix)  : float Matrix = 
    DenseMatrix.initRows mx.RowCount (fun i ->
        (mx.Row(i) - prms.Mu) ./ prms.Std
   )

//Generate ranges { [0..4], [4..8], [8..9]
//Given: rgLength = 4, seqLength = 10
let genRanges rgLength seqLength =
    seq {         
        for i in 0..rgLength..seqLength do 
            if i <> seqLength then
                yield i, if i + rgLength <= seqLength then rgLength else seqLength - i
    }


let nextGaussian (mu : float) (sigma : float) (random: System.Random)  = 
    let u1 = random.NextDouble()
    let u2 = random.NextDouble()

    let rand_std_normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2)

    mu + sigma * rand_std_normal

let nextGaussianStd : System.Random -> float = nextGaussian 0. 1. 

let swap (a: _[]) x y =
    let tmp = a.[x]
    a.[x] <- a.[y]
    a.[y] <- tmp

let permute2 (rnd: System.Random) upTo =
    let arr = [|0..upTo - 1|]
    arr |> Seq.iteri (fun i _ -> swap arr i (rnd.Next(i, upTo))) 
    arr

let permute upTo = permute2 (new System.Random()) upTo
            
(*
//Wen length = 3 and items = [1;2;3;4] -> [2;3;1]
let permute3 (rnd: System.Random) (length: int) (items : seq<_>) =
    let shuffled = generateShuffled rnd length    
    shuffled
    |> Seq.take length
    |> Seq.mapi (fun i _ -> items |> Seq.nth i)

let permute2 (len: int) (items: seq<_>) : seq<_> = 
    permute3 (new System.Random()) len items

let permute (items : seq<_>) : seq<_> = 
    permute2 (items |> Seq.length) items
*)
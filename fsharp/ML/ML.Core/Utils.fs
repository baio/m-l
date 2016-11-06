module ML.Core.Utils

open Deedle
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra
open System

let inline dprintf<'a> = sprintf "%A" >> System.Diagnostics.Debug.WriteLine    

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

let memoize f =
    let cache = ref Map.empty
    fun x ->
        match (!cache).TryFind(x) with
        | Some res -> res
        | None ->
             let res = f x
             cache := (!cache).Add(x,res)
             res            


let inline ifopt c a = if c then Some(a()) else None

let inline iif c a b = if c then a else b

let ifeq ra rb (a, b) = 
    iif (a = b) ra rb

let inline tulpe a b = a, b

let inline tulpeFlip a b = b, a

let inline split (str: string) =  str.Split(separator = [|' '|], options = StringSplitOptions.RemoveEmptyEntries)

let inline split2 (sep: string) (str: string) =  str.Split(separator = [|sep|], options = StringSplitOptions.RemoveEmptyEntries)

let inline toLower (str: string) =  str.ToLowerInvariant()
    
let inline sparse cnt ix = List.init cnt (fun i -> iif (i = ix) 1 0 |> byte)

let inline join (str: String) = String.concat
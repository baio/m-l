// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.

open Linear
open Logistic
open Softmax

[<EntryPoint>]
let main argv = 
    
    //softmax() |> ignore
    //linear() |> ignore
    logistic() |> ignore

    System.Console.ReadLine() |> ignore
    0 // return an integer exit code

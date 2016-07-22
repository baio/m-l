// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.

open Linear
open Logistic


[<EntryPoint>]
let main argv = 

    (*
    [
            [for x in 0.0 .. 0.1 .. 6.0 -> x, cos x + cos (2.0 * x)];
            [for x in 0.0 .. 0.1 .. 6.0 -> x, sin x + sin (2.0 * x)]
    ] |> showLines ["1"; "2"]
    *)

    
    linear() |> ignore
    //logistic() |> ignore

    //drawChart() |> ignore

    System.Console.ReadLine() |> ignore
    0 // return an integer exit code

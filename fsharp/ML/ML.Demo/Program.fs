// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.

open ML.Utils
open MathNet.Numerics.LinearAlgebra

[<EntryPoint>]
let main argv = 
    let mx = 
        matrix [[1.; 2.]
                [3.; 4.]]
    let t = norm mx
    printfn "%A" t
    System.Console.ReadLine() |> ignore
    0 // return an integer exit code

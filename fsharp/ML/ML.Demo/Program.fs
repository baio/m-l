// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.

open Linear
open Logistic
open Softmax
open DLinear
open DLogistic
open DSoftmax
open NN_XOR
open NN_mnist
open IO

[<EntryPoint>]
let main argv = 
    
    //comment this if MKL provider not installed
    MathNet.Numerics.Control.UseNativeMKL()        

    //logistic() |> ignore
    //softmax() |> ignore
    //linear() |> ignore
    //DLinear() |> ignore
    //DLogistic() |> ignore
    //DSoftmax() |> ignore
    //nn_xor()
    nn_mnist()
    //normalizeCsv "c:/dev/.data/mnist/mnist_train.csv" "c:/dev/.data/mnist/mnist_train_norm.csv" [0]
    //normalizeCsv "../../for_norm_test.csv" "../../for_norm_test_out.csv" [0]

    System.Console.ReadLine() |> ignore
    0 // return an integer exit code

module ML.DGD.BatchActor

open System
open Akka.Actor
open Akka.FSharp
open MathNet.Numerics.LinearAlgebra

open ML.Regressions.GradientDescent
open ML.Regressions.GD
open ML.Regressions.GLM

open Types

type BatchMessage = {
    Model: GLMModel
    BatchSize: int
    HyperParams: GradientDescentHyperParams
    Samples: BatchSamples
}

let getSamples (samples: BatchSamples) =
    match samples with
    | BatchSamples (x, y) -> (x, y)
    | _ -> failwith "not implemented"

     
let BatchActor (mailbox: Actor<BatchMessage>) = 
                   
    let rec next() = 
        actor {
    
            let! msg = mailbox.Receive()

            let _x, _y = getSamples msg.Samples
            let x = _x |> DenseMatrix.ofColumnList
            let y = _y |> DenseVector.ofList
           
            let result = gradientDescent msg.Model { EpochNumber = 1 ; ConvergeMode = ConvergeModeNone } x y msg.HyperParams

            return! next()                
        }

    next()
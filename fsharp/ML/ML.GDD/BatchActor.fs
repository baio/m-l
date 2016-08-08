module ML.DGD.BatchActor

open System
open Akka.Actor
open Akka.FSharp
open MathNet.Numerics.LinearAlgebra

open ML.Regressions.GradientDescent
open ML.Regressions.GD
open ML.Regressions.GLM

open ML.DGD.IterParamsServerActor

open Types
open SamplesStorage

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

     
let BatchActor (iterParamsServer: IActorRef) (mailbox: Actor<BatchMessage>) = 

    
    let getIterProvider (initialIterPrms) = {
        initial = (fun () -> 
            //get initial params by default, if they already exists in coordinator, they will be returned, if not will be returned initial
            let task = async { return! iterParamsServer <? SetIterParams initialIterPrms } 
            Async.RunSynchronously(task)        
        )
        update = (fun (iter) -> 
            let task = async { return! iterParamsServer <? SetIterParams(iter) } 
            Async.RunSynchronously(task)                
        )
    }
                              
    let rec next() = 
        
        actor {
    
            let! msg = mailbox.Receive()
            
            let _x, _y = getSamples msg.Samples
            let x = _x |> DenseMatrix.ofColumnList
            let y = _y |> DenseVector.ofList
            
            let iterPrr = 
                getInitialIterParams msg.Model x.ColumnCount msg.HyperParams
                |> getIterProvider
                       
            let result = gradientDescent2 iterPrr msg.Model { EpochNumber = 1 ; ConvergeMode = ConvergeModeNone } x y msg.HyperParams

            return! next()                
        }

    next()
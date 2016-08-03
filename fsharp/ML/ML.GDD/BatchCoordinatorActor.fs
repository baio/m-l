module ML.DGD.BatchCoordinatorActor

open System
open Akka.Actor
open Akka.FSharp
open MathNet.Numerics.LinearAlgebra

open ML.Regressions.GradientDescent
open ML.Regressions.GD
open ML.Regressions.GLM
open ML.Core.Utils

open BatchActor
open Types
open SamplesStorage
     
let BatchCoordinatorActor (mailbox: Actor<DGDParams>) = 
    
    // TODO : Number of children + broadcast                   
    let routerOpt = SpawnOption.Router ( Akka.Routing.FromConfig.Instance )
    let supervisionOpt = SpawnOption.SupervisorStrategy (Strategy.OneForOne(fun _ ->
            Directive.Resume
    ))

    let rec runEpoch() = 
        actor {
    
            let! msg = mailbox.Receive()

            let batchActor = spawne mailbox "BatchActor" <@ BatchActor @> [routerOpt; supervisionOpt]
           
            match msg.BatchSamples with
            | BatchSamplesProvidedByCoordinator ->
                let samples = readSamples msg.SamplesStorage None
                let batch = {
                    Model = msg.Model
                    BatchSize = msg.BatchSize
                    HyperParams = msg.HyperParams
                    Samples = BatchSamples(samples)
                }
                batchActor <! batch
                return !waitEpochComplete()
            | _ -> failwith "not implemeted"            

            return! runEpoch()                
        }
    and waitEpochComplete () =         
        actor {    
            let! msg = mailbox.Receive()
            return! waitEpochComplete()
         }

    runEpoch()
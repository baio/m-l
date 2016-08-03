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
     
let BatchCoordinatorActor (mailbox: Actor<BatchesMessage>) = 
    
    let supervisionOpt = SpawnOption.SupervisorStrategy (Strategy.OneForOne(fun _ ->
            Directive.Resume
    ))

    let rec runEpoch (cnt: int) = 
        actor {
    
            let! msg = mailbox.Receive()

            match msg with 
            | BatchesStart prms ->

                // TODO : Number of children + broadcast                   
                let routerOpt = SpawnOption.Router ( Akka.Routing.FromConfig.Instance )
                
                let batchActor = spawne mailbox "BatchActor" <@ BatchActor @> [routerOpt; supervisionOpt]
           
                match prms.BatchSamples with
                | BatchSamplesProvidedByCoordinator ->
                    let samples = readSamples prms.SamplesStorage None
                    let batch = {
                        Model = prms.Model
                        BatchSize = prms.BatchSize
                        HyperParams = prms.HyperParams
                        Samples = BatchSamples(samples)
                    }
                    batchActor <! batch
                    return! waitEpochComplete cnt prms
                | _ -> failwith "not implemeted"            

            | _ ->
                return! runEpoch(cnt + 1)                
        }
    and waitEpochComplete (cnt: int) (prms) =         
        actor {    
            let! msg = mailbox.Receive()
            match msg with 
            | BatchCompleted -> 
                if cnt < prms.EpochNumber then
                    mailbox.Self <! BatchesStart(prms)
                    return! runEpoch cnt
                // Number of epoch achieved
            | _ ->                        
                return! waitEpochComplete cnt prms
         }

    runEpoch 0
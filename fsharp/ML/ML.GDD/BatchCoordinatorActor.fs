module ML.DGD.BatchCoordinatorActor

open System
open Akka.Actor
open Akka.FSharp
open MathNet.Numerics.LinearAlgebra

open ML.Regressions.GradientDescent
open ML.Regressions.GD
open ML.Regressions.GLM
open ML.Core.Utils
open ML.Core.LinearAlgebra

open BatchActor
open Types
open SamplesStorage

let private readSamplesMem = 
    memoize (fun stg ->
        readSamples stg None
    )
     
let BatchCoordinatorActor (iterParamsServer: IActorRef) (mailbox: Actor<BatchesMessage>) = 
    

    let supervisionOpt = SpawnOption.SupervisorStrategy (Strategy.OneForOne(fun _ ->
            Directive.Escalate
    ))

    // TODO : Number of children 
    let routerOpt = SpawnOption.Router ( Akka.Routing.FromConfig.Instance )

    // spawn batch actors                
    let batchActor = spawne mailbox "BatchActor" <@ BatchActor iterParamsServer @> [routerOpt; supervisionOpt]

    //final results for each epoch
    let mutable finalResult = { ResultType = ModelTrainResultType.NaN; Theta = empty(); Errors = [] }
    //errors during epoch (returned from different batches)
    let mutable batchResults = []
    
    let rec runEpoch (epochNumber: int) = 
        actor {
    
            let! msg = mailbox.Receive()

            match msg with 
            | BatchesStart prms ->
           
                match prms.BatchSamples with
                | BatchSamplesProvidedByCoordinator ->
                    let ((x, _), y) = readSamplesMem prms.SamplesStorage
                    let batch = {
                        Model = prms.Model
                        HyperParams = prms.HyperParams
                        Samples = BatchSamples(x, y)
                    }
                    batchActor <! batch
                    return! waitEpochComplete epochNumber prms.DistributedBatchSize prms
                | _ -> failwith "not implemeted"            

            | _ ->
                return! runEpoch epochNumber               
        }
    and waitEpochComplete epochNumber batchesToComplete prms =         
        actor {    
            let! msg = mailbox.Receive()
            match msg with 
            | BatchCompleted res -> 
                batchResults <- res::batchResults
                if batchesToComplete > 1 then
                    // wait till all batches completed
                    return! waitEpochComplete epochNumber (batchesToComplete - 1) prms
                else 
                    let avgEpochError = batchResults |> List.averageBy (fun f -> f.Errors |> List.average)
                    //update final result with new theta and add avg errors of this epoch
                    finalResult <- { ResultType = res.ResultType; Theta = res.Theta; Errors = avgEpochError::finalResult.Errors  }
                    printfn "%A" finalResult
                    if epochNumber + 1 < prms.EpochNumber then                      
                        // all batches completed, next epoch 
                        //new epoch, reset batch results
                        batchResults <- [] 
                        mailbox.Self <! BatchesStart prms
                        return! runEpoch (epochNumber + 1)
                    else 
                        // all epoches completed, publish result
                        mailbox.Context.System.EventStream.Publish (BatchesCompleted(finalResult))
            | _ ->                        
                return! waitEpochComplete epochNumber batchesToComplete prms
         }

    runEpoch 0
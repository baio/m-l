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

let private readSamplesMem = memoize (fun stg -> readSamples stg None)
     
let BatchCoordinatorActor dgdParams (iterParamsServer: IActorRef) (mailbox: Actor<BatchesMessage>) = 
    
    let batch = {
        Model = dgdParams.Model
        HyperParams = dgdParams.HyperParams
        Samples = BatchSamples(emptyM(), empty())
    }

    let rowsCount = 
        match dgdParams.BatchSamples with
        | BatchSamplesProvidedByCoordinator ->
            let x, _ = readSamplesMem dgdParams.SamplesStorage
            x.RowCount
        | _ -> failwith "not implemeted"      

    let distributedBatchSize = dgdParams.DistributedBatchSize
    let batchesCnt = rowsCount / distributedBatchSize + (if rowsCount % distributedBatchSize = 0 then 0 else 1)
    
    let supervisionOpt = SpawnOption.SupervisorStrategy (Strategy.OneForOne(fun _ ->
            Directive.Escalate
    ))

    //SpawnOption.Router()

    // TODO : Number of children 
    //let conf = Akka.Routing.FromConfig.Instance

    //conf.WithFallback()
    let routerConf = new Akka.Routing.RoundRobinPool(batchesCnt)            
    let routerOpt = SpawnOption.Router ( routerConf )
         
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
                    let x, y = readSamplesMem prms.SamplesStorage
                    genRanges distributedBatchSize x.RowCount
                    |> Seq.iter(fun (start, len) ->
                        //printfn "%i %i" start len
                        let bx, by = (spliceRows start len x), (spliceVector start len y)
                        batchActor <! { batch with Samples = BatchSamples(bx, by) }
                    )                              

                    return! waitEpochComplete epochNumber batchesCnt prms
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
                    //printfn "%A" batchResults
                    let avgEpochError = batchResults |> List.averageBy (fun f -> 0.::f.Errors |> List.average)
                    //update final result with new theta and add avg errors of this epoch
                    finalResult <- { ResultType = res.ResultType; Theta = res.Theta; Errors = avgEpochError::finalResult.Errors  }                    
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
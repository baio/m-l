module ML.DGD.DistributedGradientDescnt

open Akka.Actor
open Akka.FSharp

open ML.DGD.BatchCoordinatorActor
open ML.Regressions.GD
open ML.Core.LinearAlgebra

open Types
open IterParamsServerActor

//Return async here
let DGD (prms : DGDParams) = 
    
    let system = System.create "MLDGD" (Configuration.load())
    
    let iterParamsServer = spawn system "IterParamsServerActor" (IterParamsServerActor)
    let batchCoordinator = spawn system "BatchCoordinatorActor" (BatchCoordinatorActor iterParamsServer)    
   
    let mutable dgdRes = { ResultType = NaN; Theta = empty(); Errors = [] }

    let mainActor = spawn system "main" ( actorOf2( fun mailbox msg ->
                    
        match msg with 
        | BatchesCompleted res ->
            dgdRes <- res
            system.Terminate() |> ignore
        | _ -> ()            
    ) )

    system.EventStream.Subscribe(mainActor, typedefof<BatchesMessage>) |> ignore

    batchCoordinator <! BatchesStart(prms)
        
    system.WhenTerminated.Wait()

    dgdRes

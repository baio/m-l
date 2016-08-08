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

    batchCoordinator <! prms

    let mutable dgdRes = { ResultType = NaN; Theta = zeros(0); Errors = [] }

    let mainActor = spawn system "main" ( actorOf2( fun mailbox msg ->
                    
        match msg with 
        | BatchesCompleted res ->
            dgdRes <- res
        | _ -> ()            
    ) )

    system.EventStream.Subscribe(mainActor, typedefof<BatchesMessage>) |> ignore
        
    system.WhenTerminated.Wait()

    dgdRes

module ML.DGD.DistributedGradientDescnt

open Akka.Actor
open Akka.FSharp

open ML.DGD.BatchCoordinatorActor

open Types
open IterParamsServerActor

//Return async here
let DGD (prms : DGDParams) = 
    
    let system = System.create "ML.DGD" (Configuration.load())
    
    let iterParamsServer = spawn system "IterParamsServerActor" (IterParamsServerActor)
    let batchCoordinator = spawn system "BatchCoordinator" (BatchCoordinatorActor iterParamsServer)    

    batchCoordinator <! prms
        
    system.WhenTerminated.Wait()

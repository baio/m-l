module ML.DGD.IterParamsServerActor

open System
open Akka.Actor
open Akka.FSharp
open MathNet.Numerics.LinearAlgebra

open ML.GD.GradientDescent
open ML.GD.GD
open ML.GD.GLM
open ML.Core.LinearAlgebra

open Types

//GradientDescentIter
type ThetaServerMessage = 
    | InitIterParams of obj
    //only set iter params if they still not initialzied
    | SetIterParams of obj
     
let IterParamsServerActor (mailbox: Actor<ThetaServerMessage>) = 

    let mutable latestIterParam= None
                   
    let rec next() = 
        actor {
    
            let! msg = mailbox.Receive()

            match msg with 
            | InitIterParams (iter) ->
                match latestIterParam with
                | None -> 
                    //initalize only once
                    latestIterParam <- Some(iter)
                    mailbox.Sender() <! iter
                | Some v -> 
                    mailbox.Sender() <! v
            | SetIterParams (iter) ->                
                match latestIterParam with
                | Some v ->
                    latestIterParam <- Some(iter)
                    mailbox.Sender() <! v
                | None ->
                    failwith "Params should be initialized first"
                
            return! next()                
        }

    next()
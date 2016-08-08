module ML.DGD.IterParamsServerActor

open System
open Akka.Actor
open Akka.FSharp
open MathNet.Numerics.LinearAlgebra

open ML.Regressions.GradientDescent
open ML.Regressions.GD
open ML.Regressions.GLM
open ML.Core.LinearAlgebra

open Types

//GradientDescentIter
type ThetaServerMessage<'a> = 
    //only set iter params if they still not initialzied
    | SetIterParams of 'a
     
let IterParamsServerActor (mailbox: Actor<ThetaServerMessage<'a>>) = 

    let mutable latestIterParam = None
                   
    let rec next() = 
        actor {
    
            let! msg = mailbox.Receive()

            match msg with 
            | SetIterParams iter ->                
                match latestIterParam with
                | Some v ->
                    mailbox.Sender() <! v
                | None ->
                    mailbox.Sender() <! iter
                    latestIterParam <- Some(iter)
                
            return! next()                
        }

    next()
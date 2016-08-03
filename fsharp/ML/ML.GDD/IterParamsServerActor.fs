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

type ThetaServerMessage<'a> = 
    | InitIterParams of GradientDescentIter<'a>
    | SetIterParams of GradientDescentIter<'a>
    | GetIterParams 
     
let IterParamsServerActor (mailbox: Actor<ThetaServerMessage<'a>>) = 

    let mutable latestIterParam = None
                   
    let rec next() = 
        actor {
    
            let! msg = mailbox.Receive()

            match msg with 
            | InitIterParams iter ->
                latestIterParam <- Some(iter)
            | SetIterParams iter ->
                mailbox.Sender() <! latestIterParam.Value
                latestIterParam <- Some(iter)
            | GetIterParams ->
                mailbox.Sender() <! latestIterParam.Value
                
            return! next()                
        }

    next()
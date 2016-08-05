module ML.Regressions.GradientDescent
open MathNet.Numerics.LinearAlgebra

open ML.Core.LinearAlgebra

open GLM
open GD
open SGD
open NAG
open Adagrad
open Adadelta

type GradientDescentHyperParams = 
    | SGDHyperParams of SGDHyperParams
    | NAGHyperParams of NAGHyperParams
    | AdagradHyperParams of AdagradHyperParams
    | AdadeltaHyperParams of AdadeltaHyperParams 

type GradientDescentIter' = 
    | SGDGradientDescentIter of GradientDescentIter<Unit>
    | NAGGradientDescentIter of GradientDescentIter<NAGIter>
    | AdagradGradientDescentIter of GradientDescentIter<AdagradIter>
    | AdadeltaGradientDescentIter of GradientDescentIter<AdadeltaIter>

type IterParamsProvider' = 
    | SGDIterParamsProvider of IterParamsProvider<Unit>
    | NAGIterParamsProvider of IterParamsProvider<NAGIter>
    | AdagradIterParamsProvider of IterParamsProvider<AdagradIter>
    | AdadeltaIterParamsProvider of IterParamsProvider<AdadeltaIter>

let getInitialIterParams model featuresCount hyper =    
    let _, theta, _ = getModelShapeAndTheta model featuresCount    

    match hyper with
    | SGDHyperParams hp ->                           
        SGDGradientDescentIter({ Theta = theta; Params = () })
    | NAGHyperParams hp -> 
        NAGGradientDescentIter({ Theta = theta ; Params = { Momentum = theta } })
    | AdagradHyperParams hp -> 
        AdagradGradientDescentIter({Theta = theta; Params = { G = theta }})
    | AdadeltaHyperParams hp -> 
        AdadeltaGradientDescentIter({ Theta  = theta; Params = { EG = theta; ET = theta } })

let getIterParamsProvider model featuresCount hyper  =    
    match getInitialIterParams model featuresCount hyper with
    | SGDGradientDescentIter iter ->
        SGDIterParamsProvider({ initial = (fun () -> iter) ; update =  (fun (f) -> f) })
    | NAGGradientDescentIter iter ->
        NAGIterParamsProvider({ initial = (fun () -> iter) ; update =  (fun (f) -> f) })
    | AdagradGradientDescentIter iter ->
        AdagradIterParamsProvider({ initial = (fun () -> iter) ; update =  (fun (f) -> f) })
    | AdadeltaGradientDescentIter iter ->
        AdadeltaIterParamsProvider({ initial = (fun () -> iter) ; update =  (fun (f) -> f) })
   
let gradientDescent2 
    (iterParamsProvider) 
    (model: GLMModel) 
    (prms: IterativeTrainModelParams) 
    (inputs: float Matrix) 
    (outputs: float Vector) 
    (hyper: GradientDescentHyperParams) =

    let provider = getIterParamsProvider model inputs.ColumnCount hyper

    match hyper with
    | SGDHyperParams hp ->                           
        match iterParamsProvider with 
        | SGDIterParamsProvider pr ->  
            SGD pr model prms hp inputs outputs                
        | _ -> failwith "Types of [iterParamsProvider] and [hyper] params are not consistent"
    | NAGHyperParams hp ->                           
        match iterParamsProvider with 
        | NAGIterParamsProvider pr ->  
            NAG pr model prms hp inputs outputs                
        | _ -> failwith "Types of [iterParamsProvider] and [hyper] params are not consistent"
    | AdagradHyperParams hp ->                           
        match iterParamsProvider with 
        | AdagradIterParamsProvider pr ->  
            adagrad pr model prms hp inputs outputs                
        | _ -> failwith "Types of [iterParamsProvider] and [hyper] params are not consistent"
    | AdadeltaHyperParams hp ->                           
        match iterParamsProvider with 
        | AdadeltaIterParamsProvider pr ->  
            adadelta pr model prms hp inputs outputs                
        | _ -> failwith "Types of [iterParamsProvider] and [hyper] params are not consistent"


let gradientDescent     
    (model: GLMModel) 
    (prms: IterativeTrainModelParams) 
    (inputs: float Matrix) 
    (outputs: float Vector) 
    (hyper: GradientDescentHyperParams) =
        let pr = getIterParamsProvider model inputs.ColumnCount hyper
        gradientDescent2 pr model prms inputs outputs hyper




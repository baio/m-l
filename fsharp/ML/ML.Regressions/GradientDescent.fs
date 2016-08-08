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

let getInitialIterParams<'iter> model featuresCount hyper : GradientDescentIter<'iter> =    
    let _, theta, _ = getModelShapeAndTheta model featuresCount    

    match hyper with
    | SGDHyperParams hp ->                           
        { Theta = theta; Params = box () :?> 'iter}
    | NAGHyperParams hp -> 
        { Theta = theta ; Params = box { Momentum = theta } :?> 'iter }
    | AdagradHyperParams hp -> 
        {Theta = theta; Params = box { G = theta } :?> 'iter}
    | AdadeltaHyperParams hp -> 
        { Theta  = theta; Params = box { EG = theta; ET = theta } :?> 'iter }

let getIterParamsProvider<'iter> model featuresCount hyper  =    
    match box (getInitialIterParams<'iter> model featuresCount hyper) with
    | :? GradientDescentIter<Unit> as iter ->
        { initial = (fun () -> box iter :?> GradientDescentIter<'iter>) ; update =  (fun (f) -> f) }
    | :? GradientDescentIter<NAGIter> as iter ->
        { initial = (fun () -> box iter :?> GradientDescentIter<'iter>) ; update =  (fun (f) -> f) }
    | :? GradientDescentIter<AdagradIter> as iter ->
        { initial = (fun () -> box iter :?> GradientDescentIter<'iter>) ; update =  (fun (f) -> f) }
    | :? GradientDescentIter<AdadeltaIter> as iter ->
        { initial = (fun () -> box iter :?> GradientDescentIter<'iter>) ; update =  (fun (f) -> f) }
    | _ -> failwith "Unknown type"
   
let gradientDescent2 
    (iterParamsProvider) 
    (model: GLMModel) 
    (prms: IterativeTrainModelParams) 
    (inputs: float Matrix) 
    (outputs: float Vector) 
    (hyper: GradientDescentHyperParams) =

    //let provider = getIterParamsProvider model inputs.ColumnCount hyper

    match hyper with
    | SGDHyperParams hp ->                           
        let pr = box iterParamsProvider :?> IterParamsProvider<Unit>
        SGD pr model prms hp inputs outputs                
    | NAGHyperParams hp ->                           
        let pr = box iterParamsProvider :?> IterParamsProvider<NAGIter>
        NAG pr model prms hp inputs outputs                
    | AdagradHyperParams hp ->                           
        let pr = box iterParamsProvider :?> IterParamsProvider<AdagradIter>
        adagrad pr model prms hp inputs outputs                
    | AdadeltaHyperParams hp ->                           
        let pr = box iterParamsProvider :?> IterParamsProvider<AdadeltaIter>
        adadelta pr model prms hp inputs outputs                



let gradientDescent     
    (model: GLMModel) 
    (prms: IterativeTrainModelParams) 
    (inputs: float Matrix) 
    (outputs: float Vector) 
    (hyper: GradientDescentHyperParams) =

        match hyper with
        | SGDHyperParams hp ->                           
            let pr = getIterParamsProvider<Unit> model inputs.ColumnCount hyper
            gradientDescent2 pr model prms inputs outputs hyper
        | NAGHyperParams hp ->                           
            let pr = getIterParamsProvider<NAGIter> model inputs.ColumnCount hyper
            gradientDescent2 pr model prms inputs outputs hyper
        | AdagradHyperParams hp ->                           
            let pr = getIterParamsProvider<AdagradIter> model inputs.ColumnCount hyper
            gradientDescent2 pr model prms inputs outputs hyper
        | AdadeltaHyperParams hp ->                           
            let pr = getIterParamsProvider<AdadeltaIter> model inputs.ColumnCount hyper
            gradientDescent2 pr model prms inputs outputs hyper





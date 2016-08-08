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

let getInitialIterParams model featuresCount hyper =    
    let _, theta, _ = getModelShapeAndTheta model featuresCount    

    match hyper with    
    | SGDHyperParams hp ->                           
        {Theta = theta; Params = box () :?> _}
    | NAGHyperParams hp -> 
        {Theta = theta; Params = box { Momentum = theta :?> _}}
    | AdagradHyperParams hp -> 
        {Theta = theta; Params = box { G = theta } :?> _}
    | AdadeltaHyperParams hp -> 
        { Theta  = theta; Params = box { EG = theta; ET = theta } :?> _}
    
let getIterParamsProvider model featuresCount hyper  =    
    let init = getInitialIterParams model featuresCount hyper
    { initial = (fun () -> init) ; update =  (fun (f) -> f) }

let unwrapIterParamsProvider<'iter> (iterParamsProvider: IterParamsProvider<obj>) : IterParamsProvider<'iter> =

    let init = iterParamsProvider.initial();  
    let iterInit = { Theta = init.Theta; Params = box init.Params :?> 'iter}   
    { initial = (fun () -> iterInit) ; update =  (fun (f) -> f) }
    
  
let gradientDescent2 
    (iterParamsProvider) 
    (model: GLMModel) 
    (prms: IterativeTrainModelParams) 
    (inputs: float Matrix) 
    (outputs: float Vector) 
    (hyper: GradientDescentHyperParams) =
       
    match hyper with

    | SGDHyperParams hp ->  
        let pr = unwrapIterParamsProvider<Unit>(iterParamsProvider)
        SGD pr model prms hp inputs outputs                
    | NAGHyperParams hp ->                           
        let pr = unwrapIterParamsProvider<NAGIter>(iterParamsProvider)
        NAG pr model prms hp inputs outputs                
    | AdagradHyperParams hp ->                           
        let pr = unwrapIterParamsProvider<AdagradIter>(iterParamsProvider)
        adagrad pr model prms hp inputs outputs                
    | AdadeltaHyperParams hp ->                           
        let pr = unwrapIterParamsProvider<AdadeltaIter>(iterParamsProvider)
        adadelta pr model prms hp inputs outputs                



let gradientDescent     
    (model: GLMModel) 
    (prms: IterativeTrainModelParams) 
    (inputs: float Matrix) 
    (outputs: float Vector) 
    (hyper: GradientDescentHyperParams) =
        let pr = getIterParamsProvider model inputs.ColumnCount hyper
        gradientDescent2 pr model prms inputs outputs hyper





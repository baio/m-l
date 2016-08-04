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


let gg (f: obj) : obj = f

let gradientDescent2 (iterParamsUpdate: IterParamsUpdateFunc) (model: GLMModel) (prms: IterativeTrainModelParams) (inputs: float Matrix) (outputs: float Vector) (hyper: GradientDescentHyperParams) =

    match hyper with
    | SGDHyperParams hp ->                           
        SGD2 iterParamsUpdate model prms hp inputs outputs                
    | NAGHyperParams hp -> 
        NAG2 iterParamsUpdate model prms hp inputs outputs        
    | AdagradHyperParams hp -> 
        adagrad2 iterParamsUpdate model prms hp inputs outputs        
    | AdadeltaHyperParams hp -> 
        adadelta2 iterParamsUpdate model prms hp inputs outputs        

let gradientDescent : GLMModel -> IterativeTrainModelParams -> float Matrix -> float Vector -> GradientDescentHyperParams ->  ModelTrainResult =
    gradientDescent2 (fun f -> f)


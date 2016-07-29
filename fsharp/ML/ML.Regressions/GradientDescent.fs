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

let gradientDescent (model: GLMModel) (prms: IterativeTrainModelParams) (inputs: float Matrix) (outputs: float Vector) (hyper: GradientDescentHyperParams) =
    match hyper with
    | SGDHyperParams hp ->         
        SGD model prms hp inputs outputs        
    | NAGHyperParams hp -> 
        NAG model prms hp inputs outputs        
    | AdagradHyperParams hp -> 
        adagrad model prms hp inputs outputs        
    | AdadeltaHyperParams hp -> 
        adadelta model prms hp inputs outputs        


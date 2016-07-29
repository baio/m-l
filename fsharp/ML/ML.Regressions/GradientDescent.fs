module ML.Regressions.GradientDescent
open MathNet.Numerics.LinearAlgebra

open ML.Core.LinearAlgebra

open GLM
open GD
open SGD
open NAG
open Adagrad
open Adadelta
open Theta

type GradientDescentHyperParams = 
    | SGDHyperParams of SGDHyperParams
    | NAGHyperParams of NAGHyperParams
    | AdagradHyperParams of AdagradHyperParams
    | AdadeltaHyperParams of AdadeltaHyperParams 

let inline gradientDescent (model: GLMModel) (prms: IterativeTrainModelParams) (inputs: float Matrix) (outputs: float Vector) (hyper: GradientDescentHyperParams) =
    let n = inputs.ColumnCount
    let initialTheta =  n + 1 |> zeros |> ThetaVector
    match hyper with
    | SGDHyperParams hp -> 
        let initialIter = { Theta = initialTheta ; Params = () }
        SGD initialIter model prms hp inputs outputs        
    | NAGHyperParams hp -> 
        let initialIter = { Theta = initialTheta ; Params = { Momentum = initialTheta } }
        NAG initialIter model prms hp inputs outputs        
    | AdagradHyperParams hp -> 
        let initialIter = { Theta = initialTheta ; Params = { G = initialTheta } }
        adagrad initialIter model prms hp inputs outputs        
    | AdadeltaHyperParams hp -> 
        let initialIter = { Theta = initialTheta ; Params = { EG = initialTheta ; ET = initialTheta } }
        adadelta initialIter model prms hp inputs outputs        


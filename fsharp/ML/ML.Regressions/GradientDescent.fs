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

let gradientDescentTheta (thetaShape : ThetaShape) (initialTheta: float Vector) (model: GLMModel) (prms: IterativeTrainModelParams) (inputs: float Matrix) (outputs: float Vector) (hyper: GradientDescentHyperParams) =
    match hyper with
    | SGDHyperParams hp -> 
        let initialIter = { Theta = initialTheta ; Params = () }
        SGD thetaShape initialIter model prms hp inputs outputs        
    | NAGHyperParams hp -> 
        let initialIter = { Theta = initialTheta ; Params = { Momentum = initialTheta } }
        NAG thetaShape initialIter model prms hp inputs outputs        
    | AdagradHyperParams hp -> 
        let initialIter = { Theta = initialTheta ; Params = { G = initialTheta } }
        adagrad thetaShape initialIter model prms hp inputs outputs        
    | AdadeltaHyperParams hp -> 
        let initialIter = { Theta = initialTheta ; Params = { EG = initialTheta ; ET = initialTheta } }
        adadelta thetaShape initialIter model prms hp inputs outputs        

let gradientDescent (model: GLMModel) (prms: IterativeTrainModelParams) (inputs: float Matrix) (outputs: float Vector) (hyper: GradientDescentHyperParams) =    
    let theta = 
        inputs.ColumnCount + 1 |> zeros 
    
    gradientDescentTheta ThetaShapeVector theta model prms inputs outputs hyper

let gradientDescentSoftmax (classesNumber : int) (model: GLMModel) (prms: IterativeTrainModelParams) (inputs: float Matrix) (outputs: float Vector) (hyper: GradientDescentHyperParams) =   
    let rows, cols = (inputs.ColumnCount + 1, classesNumber) 
    let theta = rows * cols |> zeros

    gradientDescentTheta (ThetaShapeMatrix(rows, cols)) theta model prms inputs outputs hyper

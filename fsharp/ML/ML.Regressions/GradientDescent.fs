module ML.Regressions.GradientDescent
open MathNet.Numerics.LinearAlgebra

open ML.Regressions.GLM
open ML.Regressions.GD
open ML.Regressions.SGD
open ML.Regressions.NAG
open ML.Regressions.Adagrad
open ML.Regressions.Adadelta

type GradientDescentHyperParams = 
    | SGDHyperParams of SGDHyperParams
    | NAGHyperParams of NAGHyperParams
    | AdagradHyperParams of AdagradHyperParams
    | AdadeltaHyperParams of AdadeltaHyperParams 

let gradientDescent (model: GLMModel) (prms: IterativeTrainModelParams) (inputs: float Matrix) (outputs: float Vector) (hyper: GradientDescentHyperParams) =
    match hyper with
    | SGDHyperParams hp -> SGD model prms hp inputs outputs        
    | NAGHyperParams hp -> NAG model prms hp inputs outputs        
    | AdagradHyperParams hp -> adagrad model prms hp inputs outputs        
    | AdadeltaHyperParams hp -> adadelta model prms hp inputs outputs        
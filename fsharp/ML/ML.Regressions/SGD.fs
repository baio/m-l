module ML.Regressions.SGD

open MathNet.Numerics.LinearAlgebra
open ML.Core.Utils
open ML.Core.LinearAlgebra
open GLM
open GD 

type SGDHyperParams = {
    Alpha: float
    BatchSize: int
}

type IterParamsUpdateFunc = obj -> obj

let calcGradientBatch<'iter, 'hyper> 
    (iterParamsUpdate: IterParamsUpdateFunc)
    (batchSize: int)
    (prms: CalcGradientParams<'hyper>)
    (iter: GradientDescentIter<'iter>) 
    (grad: ClacGradientFunc<'iter, 'hyper>) 
    =
    let x, y = permuteSamples prms.X prms.Y

    let mutable iter = iter
    genRanges batchSize x.RowCount           
    |> Seq.map (fun (start, len) -> 
        (spliceRows start len x), (spliceVector start len y)
    )
    |> Seq.iter (fun (sx, sy) ->
        iter <- (iterParamsUpdate(grad prms iter) :?> GradientDescentIter<'iter>)
    )
    iter

let private calcGradient 
    (prms: CalcGradientParams<SGDHyperParams>) 
    (iter: GradientDescentIter<Unit>) 
    =    
    let theta = iter.Theta
    let grad = iter.Theta |> prms.Gradient prms.X prms.Y
    let updatedTheta = theta - grad * prms.HyperParams.Alpha
    { Theta = updatedTheta ; Params = () }
    
let private calcGradient2 
    (iterParamsUpdate)
    (prms: CalcGradientParams<SGDHyperParams>) 
    (iter: GradientDescentIter<Unit>) 
    =
    calcGradientBatch iterParamsUpdate prms.HyperParams.BatchSize prms iter calcGradient
    
let SGD2
    (iterParamsUpdate)
    (model: GLMModel)
    (prms: IterativeTrainModelParams)    
    (hyperPrms : SGDHyperParams)
    (x : float Matrix)
    (y : float Vector) 
    =         
    let shape, theta, baseModel = getModelShapeAndTheta model x.ColumnCount    
    GD<Unit, SGDHyperParams> (calcGradient2 iterParamsUpdate) shape { Theta = theta ; Params = () } baseModel prms hyperPrms x y              

let SGD : GradientDescentFunc<SGDHyperParams> = SGD2 (fun p -> p)

module ML.Regressions.GD
// http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/

open ML.Core.LinearAlgebra
open ML.Regressions.GLM
open MathNet.Numerics.LinearAlgebra

type ConvergeMode = 
    | ConvergeModeNone
    | ConvergeModeCostStopsChange

type IterativeTrainModelParams = {
    EpochNumber : int
    ConvergeMode : ConvergeMode
}

type GradientDescentIter<'iter> = {
    Theta: float Vector     
    Params : 'iter
}

type CalcGradientParams<'hyper> = {
    HyperParams : 'hyper 
    X: float Matrix 
    Y: float Vector 
    Gradient: float Matrix -> float Vector -> float Vector -> float Vector
}

type ModelTrainResultType = Converged | MaxIterCountAchieved | NaN
type ModelTrainResult = { ResultType : ModelTrainResultType; Theta: float Vector; Errors: float list }
type ClacGradientFunc<'iter, 'hyper> = CalcGradientParams<'hyper> -> GradientDescentIter<'iter> -> GradientDescentIter<'iter>
type GradientDescentFunc<'hyper> = GLMModel -> IterativeTrainModelParams -> 'hyper -> float Matrix -> float Vector -> ModelTrainResult

let getModelShapeAndTheta (model: GLMModel) (featuresNumber: int) =
    match model with 
    | GLMBaseModel m ->
        ThetaShapeVector, featuresNumber + 1 |> zeros, m
    | GLMSoftmaxModel m ->
        let sz = featuresNumber + 1, m.ClassesNumber
        ThetaShapeMatrix(sz), fst sz * snd sz |> zeros, m.Base

let internal GD<'iter, 'hyper>    
    (calcGradient: ClacGradientFunc<'iter, 'hyper>)    
    (thetaShape: ThetaShape)
    (initialIter : GradientDescentIter<'iter>) 
    (model: GLMBaseModel)
    (prms: IterativeTrainModelParams)    
    (hyperPrms : 'hyper)
    (x : float Matrix)
    (y : float Vector) 
    : ModelTrainResult
    =

        let x = x |> appendOnes

        let calcGradientPrms = {
            HyperParams = hyperPrms
            X = x
            Y = y
            Gradient = (model.Gradient thetaShape)
        }

        let convergeCostNotImproved = 
            match prms.ConvergeMode with
            | ConvergeModeNone -> false
            | ConvergeModeCostStopsChange -> true

        let rec iterate (iter: GradientDescentIter<'iter>) errors =
            let theta = iter.Theta
            let epochCnt = errors |> List.length 
            let latestError = if errors.Length <> 0 then errors |> List.head else 0.
            let error = theta |> model.Cost thetaShape x y
            if error <> error then
                { ResultType = NaN; Theta = theta; Errors = errors }
            else if convergeCostNotImproved && latestError = error then
                // no improvements, converged
                { ResultType = Converged; Theta = theta; Errors = errors }
            else if prms.EpochNumber <= epochCnt then
                // iters count achieved
                { ResultType = MaxIterCountAchieved; Theta = theta; Errors = errors }
            else
                let updatedIterPrms = calcGradient calcGradientPrms iter
                //If graient is plus thwn we need to move down to achive function min
                //printfn "%i: \n grads: %A \n weights : %A" iterCnt gradients updatedWeights                
                iterate updatedIterPrms (error::errors) 

        iterate initialIter []

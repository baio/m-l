module ML.Regressions.GD
// http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/

open ML.Core.LinearAlgebra
open ML.Regressions.GLM
open MathNet.Numerics.LinearAlgebra

open Theta

type ConvergeMode = 
    | ConvergeModeNone
    | ConvergeModeCostStopsChange

type IterativeTrainModelParams = {
    EpochNumber : int
    ConvergeMode : ConvergeMode
}

type GradientDescentIter<'iter> = {
    Theta: Theta
    Params : 'iter
}

type CalcGradientParams<'hyper> = {
    HyperParams : 'hyper 
    X: float Matrix 
    Y: float Vector 
    Gradient: GradientFunc
}

type ModelTrainResultType = Converged | MaxIterCountAchieved
type ModelTrainResult = { ResultType : ModelTrainResultType; Theta: Theta; Errors: float list }
type ClacGradientFunc<'iter, 'hyper> = CalcGradientParams<'hyper> -> GradientDescentIter<'iter> -> GradientDescentIter<'iter>
type GradientDescentFunc<'hyper> = GLMModel -> IterativeTrainModelParams -> 'hyper -> float Matrix -> float Vector -> ModelTrainResult
//type GradientDescentFunc2<'iter, 'hyper> = GradientDescentIter<'iter> -> GLMModel -> IterativeTrainModelParams -> 'hyper -> float Matrix -> float Vector -> ModelTrainResult
//Given initial theta (all zeros) return initial iter param


let internal GD<'iter, 'hyper>    
    (calcGradient: ClacGradientFunc<'iter, 'hyper>)    
    (initialIter : GradientDescentIter<'iter>) 
    (model: GLMModel)
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
            Gradient = model.Gradient
        }

        let convergeCostNotImproved = 
            match prms.ConvergeMode with
            | ConvergeModeNone -> false
            | ConvergeModeCostStopsChange -> true

        let rec iterate (iter: GradientDescentIter<'iter>) errors =
            let theta = iter.Theta
            let epochCnt = errors |> List.length 
            let latestError = if errors.Length <> 0 then errors |> List.head else 0.
            let error = theta |> model.Cost x y
            if convergeCostNotImproved  && latestError = error then
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

module internal ML.Regressions.GradientDescent

open ML.Core.LinearAlgebra
open ML.Regressions.GLM
open MathNet.Numerics.LinearAlgebra

type GradientDescentIter<'iter> = {
    Theta: float Vector 
    Params : 'iter
}

type CalcGradientParams<'hyper> = {
    HyperParams : 'hyper 
    X: float Matrix 
    Y: float Vector 
    Gradient: GradientFunc
}

type ClacGradientFunc<'iter, 'hyper> = CalcGradientParams<'hyper> -> GradientDescentIter<'iter> -> GradientDescentIter<'iter>
type GradientDescentFunc<'hyper> = GLMModel -> IterativeTrainModelParams -> 'hyper -> float Matrix -> float Vector -> ModelTrainResult

//Given initial theta (all zeros) return initial iter param
type InitIter<'iter> = float Vector -> GradientDescentIter<'iter>

let gradientDescent<'iter, 'hyper>
    (initIter : InitIter<'iter>) 
    (calcGradient: ClacGradientFunc<'iter, 'hyper>)
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


        let rec iterate (iter: GradientDescentIter<'iter>) errors =
            let theta = iter.Theta
            let epochCnt = errors |> List.length 
            let latestError = if errors.Length <> 0 then errors |> List.head else 0.
            let error = model.Loss theta x y
            if latestError = error then
                // no improvements, converged
                { ResultType = Converged; Weights = theta; Errors = errors }
            else if error <= prms.MinErrorThreshold then
                // got minimal error threshold
                { ResultType = ErrorThresholdAchieved; Weights = theta; Errors = errors }
            else if prms.EpochNumber < epochCnt then
                // iters count achieved
                { ResultType = MaxIterCountAchieved; Weights = theta; Errors = errors }
            else
                let updatedIterPrms = calcGradient calcGradientPrms iter
                //If graient is plus thwn we need to move down to achive function min
                //printfn "%i: \n grads: %A \n weights : %A" iterCnt gradients updatedWeights                
                iterate updatedIterPrms (error::errors) 

        // initialize random weights
        let initialTheta = x.ColumnCount |> zeros 
        iterate (initIter initialTheta) []

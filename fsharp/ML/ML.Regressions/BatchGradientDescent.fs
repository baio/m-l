module ML.Regressions.BatchGradientDescent

open ML.Core.LinearAlgebra
open ML.Regressions.GLM
open MathNet.Numerics.LinearAlgebra
  
let batchGradientDescent
    (model: GLMModel)
    (prms: IterativeTrainModelParams)
    (x : float Matrix)
    (y : float Vector) =

        let x = x |> appendOnes

        let rec iter w errors =
            let iterCnt = errors |> List.length 
            let latestError = if errors.Length <> 0 then errors |> List.head else 0.
            let error = model.Loss w x y
            if latestError = error then
                // no improvements, converged
                { ResultType = Converged; Weights = w; Errors = errors }
            else if error <= prms.MinErrorThreshold then
                // got minimal error threshold
                { ResultType = ErrorThresholdAchieved; Weights = w; Errors = errors }
            else if prms.MaxIterNumber < iterCnt then
                // iters count achieved
                { ResultType = MaxIterCountAchieved; Weights = w; Errors = errors }
            else
                let gradients = model.Gradient w x y
                //If graient is plus thwn we need to move down to achive function min
                let theta = w - prms.Alpha * gradients
                //printfn "%i: \n grads: %A \n weights : %A" iterCnt gradients updatedWeights                
                iter theta (error::errors)

        // initialize random weights
        let initialW = x.ColumnCount |> zeros 
        iter initialW []

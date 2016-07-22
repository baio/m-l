module ML.Regressions.GradientDescent

open ML.Core.LinearAlgebra
open ML.Regressions.GLM
open MathNet.Numerics.LinearAlgebra

let gradientDescent
    (weightsCalculator: WeightsCalculator)
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
                Converged, w, errors
            else if error <= prms.MinErrorThreshold then
                // got minimal error threshold
                ErrorThresholdAchieved, w, errors
            else if prms.MaxIterNumber < iterCnt then
                // iters count achieved
                MaxIterCountAchieved, w, errors
            else
                let theta = weightsCalculator w x y
                //If graient is plus thwn we need to move down to achive function min
                //printfn "%i: \n grads: %A \n weights : %A" iterCnt gradients updatedWeights                
                iter theta (error::errors)

        // initialize random weights
        let initialW = x.ColumnCount |> zeros 
        iter initialW []

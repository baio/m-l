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

        let rec iter w iterCnt latestError =
            let error = model.Loss w x y
            if latestError = error then
                // no improvements, converged
                Converged, w
            else if error <= prms.MinErrorThreshold then
                // got minimal error threshold
                ErrorThresholdAchieved, w
            else if prms.MaxIterNumber < iterCnt then
                // iters count achieved
                MaxIterCountAchieved, w
            else
                let gradients = model.Gradient w x y
                //If graient is plus thwn we need to move down to achive function min
                let theta = w - prms.Alpha * gradients
                //printfn "%i: \n grads: %A \n weights : %A" iterCnt gradients updatedWeights
                iter theta (iterCnt + 1) error

        // initialize random weights
        let initialW = x.ColumnCount |> zeros 
        iter initialW 0 0.

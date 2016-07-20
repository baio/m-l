module ML.Regressions.MiniBatchGradientDescent

open ML.Core.Utils
open ML.Core.LinearAlgebra
open ML.Regressions.GLM
open MathNet.Numerics.LinearAlgebra

let miniBatchGradientDescent
    (model: GLMModel)
    (prms: MiniBatchTrainModelParams)
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
                let mutable theta = w 
                genRanges prms.BatchSize x.RowCount           
                |> Seq.map (fun (start, len) -> 
                    (spliceRows start len x), (spliceVector start len y)
                )
                |> Seq.iter (fun (sx, sy) ->
                    let gradients = model.Gradient theta sx sy
                    theta <- theta - prms.Alpha * gradients                
                )
                iter theta (iterCnt + 1) error

        // initialize random weights
        let initialW = x.ColumnCount |> zeros 
        iter initialW 0 0.

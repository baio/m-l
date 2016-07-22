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
                let mutable theta = w 
                genRanges prms.BatchSize x.RowCount           
                |> Seq.map (fun (start, len) -> 
                    (spliceRows start len x), (spliceVector start len y)
                )
                |> Seq.iter (fun (sx, sy) ->
                    let gradients = model.Gradient theta sx sy
                    theta <- theta - prms.Alpha * gradients                
                )
                iter theta (error::errors)

        // initialize random weights
        let initialW = x.ColumnCount |> zeros 
        iter initialW []

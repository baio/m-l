module ML.Regressions.Adadelta


open ML.Core.Utils
open ML.Core.LinearAlgebra
open ML.Regressions.GLM
open MathNet.Numerics.LinearAlgebra

let adagradGradientDescent
    (model: GLMModel)
    (prms: AdagradTrainModelParams)
    (x : float Matrix)
    (y : float Vector) =

        let x = x |> appendOnes

        let rec iter w errors (G: float Vector) =
            let epochCnt = errors |> List.length 
            let latestError = if errors.Length <> 0 then errors |> List.head else 0.    
            let error = model.Loss w x y
            if latestError = error then
                // no improvements, converged
                { ResultType = Converged; Weights = w; Errors = errors }
            else if error <= prms.MinErrorThreshold then
                // got minimal error threshold
                { ResultType = ErrorThresholdAchieved; Weights = w; Errors = errors }
            else if prms.EpochNumber < epochCnt then
                // iters count achieved
                { ResultType = MaxIterCountAchieved; Weights = w; Errors = errors }
            else    
                let mutable g = G
                let mutable theta = w 
                genRanges prms.BatchSize x.RowCount           
                |> Seq.map (fun (start, len) -> 
                    (spliceRows start len x), (spliceVector start len y)
                )
                |> Seq.iter (fun (sx, sy) ->                    
                    let gradients = model.Gradient theta sx sy
                    let k = prms.Alpha / (prms.Epsilon + g.PointwisePower(0.5))                    
                    theta <- theta - k .* gradients                
                    g <- g + gradients.PointwisePower(2.)
                )
                iter theta (error::errors) g

        // initialize random weights
        let initialW = x.ColumnCount |> zeros 
        let initialG = x.ColumnCount |> ones
        iter initialW [] initialG

module ML.Regressions.NesterovAcceleratedGradient

open ML.Core.Utils
open ML.Core.LinearAlgebra
open ML.Regressions.GLM
open MathNet.Numerics.LinearAlgebra

let nesterovAcceleratedGradientDescent
    (model: GLMModel)
    (prms: AcceleratedTrainModelParams)
    (x : float Matrix)
    (y : float Vector) =

        let x = x |> appendOnes

        let rec iter w errors (latestMomentum: float Vector) =
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
                let mutable theta = w
                let mutable momentum = latestMomentum
                genRanges prms.BatchSize x.RowCount           
                |> Seq.map (fun (start, len) -> 
                    (spliceRows start len x), (spliceVector start len y)
                )
                |> Seq.iter (fun (sx, sy) ->                    
                    let accelaration = prms.Gamma * momentum
                    let gradients = model.Gradient (theta - accelaration) sx sy
                    momentum <- accelaration + prms.Alpha * gradients
                    theta <- theta - momentum
                )
                iter theta (error::errors) momentum

        // initialize random weights
        let initialW = x.ColumnCount |> zeros 
        iter initialW [] initialW 

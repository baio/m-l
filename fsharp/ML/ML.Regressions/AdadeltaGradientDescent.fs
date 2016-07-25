// http://arxiv.org/pdf/1212.5701v1.pdf
// http://climin.readthedocs.io/en/latest/adadelta.html
module ML.Regressions.AdadeltaGradientDescent

open ML.Core.Utils
open ML.Core.LinearAlgebra
open ML.Regressions.GLM
open MathNet.Numerics.LinearAlgebra

let adadeltaGradientDescent
    (model: GLMModel)
    (prms: AdadeltaTrainModelParams)
    (x : float Matrix)
    (y : float Vector) =

        let x = x |> appendOnes

        let rec iter w errors (Eg: float Vector) (Et: float Vector) =
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
                let mutable eg = Eg
                let mutable et = Et
                let mutable theta = w 
                genRanges prms.BatchSize x.RowCount           
                |> Seq.map (fun (start, len) -> 
                    (spliceRows start len x), (spliceVector start len y)
                )
                |> Seq.iter (fun (sx, sy) ->                  
                    //calculate gradient  
                    let gradients = model.Gradient theta sx sy
                    //accumulate gradient
                    eg <- prms.Rho * eg + (1. - prms.Rho) * gradients.PointwisePower(2.)                    
                    //compute update
                    let rms_t = (et + prms.Epsilon).PointwisePower(0.5)
                    let rms_g = (eg + prms.Epsilon).PointwisePower(0.5)
                    let d_theta = (rms_t / rms_g) .* gradients
                    //accumulate updates
                    et <- prms.Rho * et + (1. - prms.Rho) * d_theta.PointwisePower(2.)                    
                    //apply update
                    theta <- theta - d_theta
                )
                iter theta (error::errors) eg et

        // initialize random weights
        let initialW = x.ColumnCount |> zeros 
        let initialG = x.ColumnCount |> ones
        iter initialW [] initialW initialW

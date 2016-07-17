module ML.Regressions.StochasticGradientDescent

open ML.Utils
open ML.Regressions.GLM
open MathNet.Numerics.LinearAlgebra

// returns true, weights - if error threshold achieved
// fales, weights - if max number of iterations achieved
let stochasticGradientDescent
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
                let mutable theta = w            
                x |> Matrix.iteriRows (fun i _ ->
                    let sx = x |> spliceRows i 1 
                    let sy = y |> spliceVector i 1
                    let gradients = model.Gradient w sx sy
                    theta <- theta - prms.Alpha * gradients                
                )
                iter theta (iterCnt + 1) error

        // initialize random weights
        let initialW = x.ColumnCount |> zeros 
        iter initialW 0 0.

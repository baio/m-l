﻿module ML.Regressions.NesterovAcceleratedGradient

open ML.Utils
open ML.Regressions.GLM
open MathNet.Numerics.LinearAlgebra

// returns true, weights - if error threshold achieved
// fales, weights - if max number of iterations achieved
let nesterovAcceleratedGradientDescent
    (model: GLMModel)
    (prms: AcceleratedTrainModelParams)
    (x : float Matrix)
    (y : float Vector) =

        let x = x |> appendOnes

        let rec iter w epochCnt (latestMomentum: float Vector) latestError =
            let error = model.Loss w x y
            if latestError = error then
                // no improvements, converged
                Converged, w
            else if error <= prms.MinErrorThreshold then
                // got minimal error threshold
                ErrorThresholdAchieved, w
            else if prms.EpochNumber < epochCnt then
                // iters count achieved
                MaxIterCountAchieved, w
            else    
                let accelaration = prms.Gamma * latestMomentum
                let mutable theta = w
                let mutable momentum = latestMomentum
                genRanges prms.BatchSize x.RowCount           
                |> Seq.map (fun (start, len) -> 
                    (spliceRows start len x), (spliceVector start len y)
                )
                |> Seq.iter (fun (sx, sy) ->                    
                    let gradients = model.Gradient (theta - accelaration) sx sy
                    momentum <- accelaration + prms.Alpha * gradients
                    theta <- theta - momentum
                )
                iter theta (epochCnt + 1) momentum error

        // initialize random weights
        let initialW = x.ColumnCount |> zeros 
        iter initialW 0 initialW 0. 
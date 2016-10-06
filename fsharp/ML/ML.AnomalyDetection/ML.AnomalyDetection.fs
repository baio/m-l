module ML.AnomalyDetection.AnomalyDetection

open System
open MathNet.Numerics.LinearAlgebra

open ML.Core.LinearAlgebra
open Deedle

type AnomalyModel = { mu: float[]; std : float[] }

let inline toArray (series: Series<'a, 'b>) = series |> Series.valuesAll |> Seq.choose(fun f -> f) |> Seq.toArray
    
let calcGauss mu std x = 
    (1. / Math.Sqrt(2. * Math.PI) * std) * Math.Exp(-1. * Math.Pow((x - mu), 2.) / (2. * Math.Pow(std, 2.)))

let calcAnomalyModel (mx: FMatrix) = 
    let frame = mx |> Matrix.toArray2 |> Frame.ofArray2D    
    {
        mu = frame |> Stats.mean |> toArray  
        std = frame |> Stats.stdDev |> toArray
    }
    
let calcP (model: AnomalyModel) (mx: FMatrix) =
    mx 
    |> Matrix.mapCols (fun i ->         
        Vector.map (calcGauss model.mu.[i] model.std.[i])
    ) 
    |> Matrix.toRowSeq
    |> Seq.map (Vector.fold (fun acc v -> acc * v) 1.)
    

//return vector of elements where for each matrix row true is anomaly and false is not
let findAnomalies (model: AnomalyModel) (mx: FMatrix) (epsilon: float) =
    calcP model mx
    |> Seq.map(fun f -> f >= epsilon)    



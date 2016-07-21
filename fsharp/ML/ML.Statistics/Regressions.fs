module ML.Statistics.Regressions
// http://stattrek.com/regression/slope-test.aspx?Tutorial=AP
// https://msdn.microsoft.com/en-us/magazine/mt620016.aspx
// https://www.khanacademy.org/math/probability/regression/regression-correlation/v/calculating-r-squared
open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Statistics

open ML.Statistics
open ML.Statistics.StudentT

open ML.Core.LinearAlgebra
open ML.Core.Utils

type NormParams = {Mu : float; Std: float}
    
let regressionDF (paramsNumber: int) = paramsNumber - 2

let regressionSE (x: float seq) (y: float seq) = 
    let pow2 v = Math.Pow(v, 2.)
    let x_m = x |> Statistics.Mean
    let y_m = y |> Statistics.Mean
    Math.Sqrt(y |> Seq.sumBy (fun yi -> pow2 <| yi - y_m)) / (x |> Seq.length |> regressionDF |> float) /
        Math.Sqrt(x |> Seq.sumBy (fun xi -> pow2 <| xi - x_m))

let regressionT (x: float seq) (y: float seq) (coefficent: float) =     
    coefficent - (regressionSE x y)
    
let regressionP (x: float seq) (y: float seq) (coefficent: float) = 
    regressionT x y coefficent
    |> studentT (Seq.length(x) - 2)

let wholeRegressionP (x: float Matrix) (y: float Vector) (coefficents: float Vector) = 
    (x |> appendOnes).EnumerateColumns()
    |> Seq.mapi (fun i xcol -> regressionP xcol y (coefficents.At i))
    
let regressionRSquared (normX: float Matrix) (y: float Vector) (w: float Vector) =     
    let y_m = y |> Statistics.Mean
    let yErrSq = (y - y_m).PointwisePower(2.).Sum()    
    let x = normX |> appendOnes 
    let h = x * w
    let hErrSq = (y - h).PointwisePower(2.).Sum()
    1. - (hErrSq / yErrSq)    
    
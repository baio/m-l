module ML.Utils
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra

let norm (mx: float Matrix) = 
    mx.NormalizeColumns 1.
    
module ML.Regressions.SoftmaxRegression

open MathNet.Numerics.LinearAlgebra
open ML.Core.Utils
open ML.Core.LinearAlgebra

open GLM

let softmaxHyp (theta: float Matrix) (x: float Vector) = 
    let sum = (theta * x).PointwiseExp() |> Vector.sum 
    let exps = (theta .* x.ToColumnMatrix()).PointwiseExp()
    (1. / sum) * exps

let private softmax (theta: float Matrix) (x: float Matrix) =     
    // n = number of features
    // m = numbers of samples in x
    // k = number of classes
    // -------------------------
    // theta : n * k
    // x : n * m    
    //---------------------------
    // Returns : m * k
    
    // sum : m * 1
    let sum = (x.Transpose() * theta).PointwiseExp().ColumnSums().ToRowMatrix()
    // sumrep : m * k (all column values for each row are the same)
    let sumrep = sum |> repmatCol theta.ColumnCount
    // exps : m * k
    let exps = (x.Transpose() * theta).PointwiseExp()
    
    sumrep ./ exps


let softmaxCost (theta: float Matrix) (x : float Matrix) (y : float Vector) = 
    // n = number of features
    // m = numbers of samples in x
    // k = number of classes
    // -------------------------
    // theta : n * k
    // x : n * m    
    //---------------------------
    //Returns 1 * k

    // p, lp : m * k
    let logP = (softmax theta x).PointwiseLog()
    
    // mi : m * k
    // one hot ?
    // in each row only one column with index equal a class number is 1, allothers are zero
    let oneHot = encodeOneHot theta.ColumnCount y 

    // m * k : then sum all
    oneHot .* logP |> Matrix.sumCols
    

let softmaxGradient (theta: float Matrix) (x: float Matrix) (y: float Vector) = 
    let oneHot = encodeOneHot theta.ColumnCount y 
    let p = softmax theta x
    x * (oneHot - p) |> Matrix.sumCols |> (*) -1.


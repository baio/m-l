module ML.Regressions.SoftmaxRegression

open MathNet.Numerics.LinearAlgebra
open ML.Core.Utils
open ML.Core.LinearAlgebra

open GLM
open Theta

let softmaxHyp (thetaShape: ThetaShape) (x: float Vector) (_theta: float Vector) = 
    let theta = reshape (thetaShape.matrixSize()) _theta
    let sum = (theta * x).PointwiseExp() |> Vector.sum 
    let exps = (theta .* x.ToColumnMatrix()).PointwiseExp()
    (1. / sum) * exps 

let private softmax (x: float Matrix) (theta: float Matrix) =     

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


let softmaxCost (thetaShape: ThetaShape) (x : float Matrix) (y : float Vector) (_theta: float Vector) = 
    
    let theta = reshape (thetaShape.matrixSize()) _theta
    // n = number of features
    // m = numbers of samples in x
    // k = number of classes
    // -------------------------
    // theta : n * k
    // x : n * m    
    //---------------------------
    //Returns 1 * k

    // p, lp : m * k
    let logP = (softmax x theta).PointwiseLog()
    
    // mi : m * k
    // in each row only one column with index equal a class number is 1, allothers are zero
    let oneHot = encodeOneHot theta.ColumnCount y 

    // m * k : then sum all 
    oneHot .* logP |> Matrix.sum |> (*) -1.
        

let softmaxGradient (thetaShape: ThetaShape) (x: float Matrix) (y: float Vector) (_theta: float Vector) = 
    let theta = reshape (thetaShape.matrixSize()) _theta
    //Returns n * k
    // m * k
    let oneHot = encodeOneHot theta.ColumnCount y 
    // m * k
    let p = softmax theta x
    // x : n * m
    x .* (oneHot - p) |> (*) -1. |> flat
    


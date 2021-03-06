﻿module ML.GD.SoftmaxRegression

open MathNet.Numerics.LinearAlgebra
open ML.Core.Utils
open ML.Core.LinearAlgebra

open GLM

let private softmax (x: float Matrix) (theta: float Matrix) =     

    // n = number of features
    // m = numbers of samples in x
    // k = number of classes
    // -------------------------
    // theta : n * k
    // x : m  * n
    //---------------------------
    // Returns : m * k
    
    // exps : m * k
    let exps = (x * theta).PointwiseExp()
    // exp : m * 1
    let sum = exps.RowSums().ToColumnMatrix()

    // sumrep : m * k (all column values for each row are the same)
    let sumrep = sum |> repmatCol theta.ColumnCount
    
    exps ./ sumrep


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

    // lp : m * k
    let logP = (softmax x theta).PointwiseLog()
    
    // mi : m * k
    // in each row only one column with index equal a class number is 1, allothers are zero
    let oneHot = encodeOneHot theta.ColumnCount y 

    // m * k : then sum all 
    let a = oneHot .* logP 
    (a |> Matrix.sum |> (*) -1.) / float x.RowCount
        

let softmaxGradient (thetaShape: ThetaShape) (x: float Matrix) (y: float Vector) (_theta: float Vector) = 
    let theta = reshape (thetaShape.matrixSize()) _theta
    //Returns n * k
    // m * k
    let oneHot = encodeOneHot theta.ColumnCount y 
    // m * k
    let p = softmax x theta 

    x.Transpose() * (oneHot - p) / float x.RowCount |> (*) -1. |> flat
   
// number of classes, theta, x, y
let predict (classesNumber: int) (_x: float Matrix) (_theta: float Vector) : float Vector =         
    let x = _x |> appendOnes
    let theta = reshape (x.ColumnCount, classesNumber) _theta
    let s = softmax x theta     
    s.EnumerateRows()
    |> Seq.map (fun v -> v.MaximumIndex() |> float)
    |> DenseVector.ofSeq
        
// number of classes, theta, x, y
let accuracy (classesNumber: int) (x: float Matrix) (y: float Vector) (theta: float Vector) : float = 
    predict classesNumber x theta
    |> GLMCorrectPercent y
    

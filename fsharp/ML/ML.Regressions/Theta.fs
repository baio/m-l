module ML.Regressions.Theta

open MathNet.Numerics.LinearAlgebra

type Theta = 
    | ThetaVector of float Vector // for basic cases, Linear, Logistsic when theta is a Vector n
    | ThetaMatrix of float Matrix // for softmax when theta is a Matrix K * n
    member this.asVector()  =
        match this with
        | ThetaVector vec -> vec
        | _ -> failwith "theta is not a vector"
    member this.asMatrix() =
        match this with
        | ThetaMatrix mx -> mx
        | _ -> failwith "theta is not a matrix"
    static member (*) (t: Theta, f : float) =
        match t with
        | ThetaVector vec -> ThetaVector(vec * f)
        | ThetaMatrix mx -> ThetaMatrix(mx * f)
    static member (-) (t: Theta, v : float Vector) =
        match t with
        | ThetaVector vec -> ThetaVector(vec - v)
        | _ -> failwith "TODO" //ThetaMatrix mx -> ThetaMatrix(mx - f)
    static member (+) (t: Theta, v : float Vector) =
        match t with
        | ThetaVector vec -> ThetaVector(vec + v)
        | _ -> failwith "TODO" //ThetaMatrix mx -> ThetaMatrix(mx - f)
    static member (+) (t: Theta, s : float) =
        match t with
        | ThetaVector vec -> ThetaVector(vec + s)
        | ThetaMatrix mx -> ThetaMatrix(mx + s)
    static member ( .* ) (t: Theta, v : float Vector) =
        match t with
        | ThetaVector vec -> ThetaVector(vec .* v)
        | _ -> failwith "TODO" //ThetaMatrix mx -> ThetaMatrix(mx - f)
    static member (.^) (t: Theta, s : float) =
        match t with
        | ThetaVector vec -> ThetaVector(vec.PointwisePower(s))
        | ThetaMatrix mx -> ThetaMatrix(mx.PointwisePower(s))
    static member (-) (t1 : Theta, t2: Theta) =
        match t1, t2 with
        | ThetaVector(vec1), ThetaVector(vec2) -> 
           ThetaVector(vec1 - vec2)
        | ThetaMatrix(mx1), ThetaMatrix(mx2) -> 
            ThetaMatrix(mx1 - mx2)
        | _ -> failwith "Types of thetas are different"
    static member (+) (t1 : Theta, t2: Theta) =
        match t1, t2 with
        | ThetaVector(vec1), ThetaVector(vec2) -> 
           ThetaVector(vec1 + vec2)
        | ThetaMatrix(mx1), ThetaMatrix(mx2) -> 
            ThetaMatrix(mx1 + mx2)
        | _ -> failwith "Types of thetas are different"
    static member (/) (t1 : Theta, t2: Theta) =
        match t1, t2 with
        | ThetaVector(vec1), ThetaVector(vec2) -> 
            ThetaVector(vec1 ./ vec2)
        | ThetaMatrix(mx1), ThetaMatrix(mx2) -> 
            ThetaMatrix(mx1 ./ mx2)
        | _ -> failwith "Types of thetas are different"
    static member ( .* ) (t1: Theta, t2 : Theta) =
        match t1, t2 with
        | ThetaVector(vec1), ThetaVector(vec2) -> 
            ThetaVector(vec1 .* vec2)
        | ThetaMatrix(mx1), ThetaMatrix(mx2) -> 
            ThetaMatrix(mx1 .* mx2)
        | _ -> failwith "Types of thetas are different"
    static member ( / ) (s: float, t : Theta) =
        match t with
        | ThetaVector vec -> ThetaVector(s / vec)
        | ThetaMatrix mx ->  ThetaMatrix(s / mx)
                                               
type ThetaVectorBuilder() =     

    member this.Bind(m, f)  =
        match m with 
            | ThetaVector vec -> f vec
            | _ -> failwith "theta is not vector"             
    
    member this.Return (x) = 
        ThetaVector(x)
                

let thetaVector = new ThetaVectorBuilder()
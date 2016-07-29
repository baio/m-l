module ML.Regressions.Theta

open MathNet.Numerics.LinearAlgebra

open ML.Core.LinearAlgebra

type Theta = 
    | ThetaVector of float Vector // for basic cases, Linear, Logistsic when theta is a Vector n
    | ThetaMatrix of float Matrix // for softmax when theta is a Matrix K * n
    member this.vector()  =
        match this with
        | ThetaVector vec -> vec
        | _ -> failwith "theta is not a vector"
    member this.matrix() =
        match this with
        | ThetaMatrix mx -> mx
        | _ -> failwith "theta is not a matrix"
    member this.asVector() =
        match this with
        | ThetaVector vec -> vec
        | ThetaMatrix mx -> mx |> flat
    member this.fromVector(vec: float Vector) =
        match this with
        | ThetaVector vec -> ThetaVector(vec)
        | ThetaMatrix mx -> 
            ThetaMatrix(reshape (mx.RowCount, mx.ColumnCount) vec)

(*
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
    //some math
    member this.sum() = 
        match this with
        | ThetaVector vec -> vec.Sum()
        | ThetaMatrix mx -> mx |> Matrix.sum
    member this.log() = 
        match this with
        | ThetaVector vec -> ThetaVector(vec.PointwiseLog())
        | ThetaMatrix mx -> ThetaMatrix(mx.PointwiseLog())
    member this.exp() = 
        match this with
        | ThetaVector vec -> ThetaVector(vec.PointwiseExp())
        | ThetaMatrix mx -> ThetaMatrix(mx.PointwiseExp())
    //Theta & Scalars            
    static member (.^) (t: Theta, s : float) =
        match t with
        | ThetaVector vec -> ThetaVector(vec.PointwisePower(s))
        | ThetaMatrix mx -> ThetaMatrix(mx.PointwisePower(s))
    static member (*) (f : float, t: Theta) =
        match t with
        | ThetaVector vec -> ThetaVector(vec * f)
        | ThetaMatrix mx -> ThetaMatrix(mx * f)
    static member (*) (t: Theta, f : float) =
        match t with
        | ThetaVector vec -> ThetaVector(vec * f)
        | ThetaMatrix mx -> ThetaMatrix(mx * f)
    static member (+) (f : float, t: Theta) =
        match t with
        | ThetaVector vec -> ThetaVector(vec + f)
        | ThetaMatrix mx -> ThetaMatrix(mx + f)
    static member (+) (t: Theta, f : float) =
        match t with
        | ThetaVector vec -> ThetaVector(vec + f)
        | ThetaMatrix mx -> ThetaMatrix(mx + f)
    static member (-) (f : float, t: Theta) =
        match t with
        | ThetaVector vec -> ThetaVector(f - vec)
        | ThetaMatrix mx -> ThetaMatrix(f - mx)
    static member (-) (t: Theta, f : float) =
        match t with
        | ThetaVector vec -> ThetaVector(vec - f)
        | ThetaMatrix mx -> ThetaMatrix(mx - f)
    static member ( / ) (t : Theta, s: float) =
        match t with
        | ThetaVector vec -> ThetaVector(vec / s)
        | ThetaMatrix mx ->  ThetaMatrix(mx / s)
    static member ( / ) (s: float, t : Theta) =
        match t with
        | ThetaVector vec -> ThetaVector(s / vec)
        | ThetaMatrix mx ->  ThetaMatrix(s / mx)
    //Theta & Vectors            
    static member (-) (t: Theta, v : float Vector) =
        match t with
        | ThetaVector vec -> ThetaVector(vec - v)
        | _ -> failwith "TODO" //ThetaMatrix mx -> ThetaMatrix(mx - f)
    static member (+) (t: Theta, v : float Vector) =
        match t with
        | ThetaVector vec -> ThetaVector(vec + v)
        | _ -> failwith "TODO" //ThetaMatrix mx -> ThetaMatrix(mx - f)
    static member ( .* ) (t: Theta, v : float Vector) =
        match t with
        | ThetaVector vec -> ThetaVector(vec .* v)
        | _ -> failwith "TODO" //ThetaMatrix mx -> ThetaMatrix(mx - f)
    static member ( * ) (t : Theta, v : float Vector) =
        match t with
        | ThetaVector vec -> vec * v
        | ThetaMatrix mx ->  (mx * v).Maximum()
    static member ( * ) (v : float Vector, t : Theta) =
        match t with
        | ThetaVector vec -> vec * v
        | ThetaMatrix mx ->  (mx * v).Maximum()
    //Theta & Matrix
    static member ( * ) (m: float Matrix, t : Theta) =
        match t with
        | ThetaVector vec -> ThetaVector(m * vec)
        | ThetaMatrix mx ->  ThetaMatrix(m * mx)
    //Theta & Theta
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
                                               
type ThetaVectorBuilder() =     

    member this.Bind(m, f)  =
        match m with 
            | ThetaVector vec -> f vec
            | _ -> failwith "theta is not vector"             
    
    member this.Return (x) = 
        ThetaVector(x)
                

let thetaVector = new ThetaVectorBuilder()
*)
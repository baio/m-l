module ML.GradientCheck
open MathNet.Numerics.LinearAlgebra

open ML.Core.LinearAlgebra
open ML.Core.Utils
open ML.NN

let private _J (t: FMatrix) (y: FMatrix) = (t - y).PointwisePower(2.) / 2.

let gradCheck (target: FVector) (inputs: FVector) shape (theta: FVector) epsilon = 
    let imx = DenseMatrix.ofRowSeq [inputs]
    let tmx = DenseMatrix.ofRowSeq [target]
    theta |> Vector.mapi (fun c col ->
        let ltheta = theta.MapIndexed(fun i x -> iif (i = c) (x + epsilon) x)
        let rtheta = theta.MapIndexed(fun i x -> iif (i = c) (x - epsilon) x)
        let res1 = forwardOutput imx shape ltheta |> _J tmx
        let res2 = forwardOutput imx shape rtheta |> _J tmx
        let v1 = res1.Row(0).At(0)
        let v2 = res2.Row(0).At(0)
        (v1 - v2) / (2. * epsilon)                        
    )





    

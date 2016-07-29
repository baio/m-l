module ML.Regressions.SGD

open MathNet.Numerics.LinearAlgebra
open ML.Core.Utils
open ML.Core.LinearAlgebra
open Theta
open GLM
open GD 

type SGDHyperParams = {
    Alpha: float
    BatchSize: int
}

let calcGradientBatch<'iter, 'hyper> (batchSize: int) (prms: CalcGradientParams<'hyper>) (iter: GradientDescentIter<'iter>) (grad: ClacGradientFunc<'iter, 'hyper>) =
    let x = prms.X
    let y = prms.Y

    let mutable iter = iter
    genRanges batchSize x.RowCount           
    |> Seq.map (fun (start, len) -> 
        (spliceRows start len x), (spliceVector start len y)
    )
    |> Seq.iter (fun (sx, sy) ->
        iter <- grad prms iter        
    )
    iter

let private calcGradient (prms: CalcGradientParams<SGDHyperParams>) (iter: GradientDescentIter<Unit>) =    
    let theta = iter.Theta
    let grad = iter.Theta |> prms.Gradient prms.X prms.Y
    let updatedTheta = theta - grad * prms.HyperParams.Alpha
    { Theta = updatedTheta ; Params = () }
    
let private calcGradient2 (prms: CalcGradientParams<SGDHyperParams>) (iter: GradientDescentIter<Unit>) =
    calcGradientBatch prms.HyperParams.BatchSize prms iter calcGradient
    
let SGD (thetaShape: ThetaShape) (initialIter : GradientDescentIter<Unit>) : GradientDescentFunc<SGDHyperParams> = 
    GD<Unit, SGDHyperParams> calcGradient2 thetaShape initialIter

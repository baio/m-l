module ML.Regressions.BatchGradientDescent

open ML.Core.LinearAlgebra
open ML.Regressions.GLM
open MathNet.Numerics.LinearAlgebra
open GradientDescent 

let calcGradient (prms: CalcGradientParams<BasicHyperParams>) (iter: GradientDescentIter<Unit>) =
    let theta = iter.Theta
    let gradients = prms.Gradient theta prms.X prms.Y
    { Theta  = theta - prms.HyperParams.Alpha * gradients; Params = ()}

let initIter (initialTheta: float Vector) = { Theta  = initialTheta; Params = () }
    
let batchGradientDescent : GradientDescentFunc<BasicHyperParams> = 
    gradientDescent<Unit, BasicHyperParams> initIter calcGradient


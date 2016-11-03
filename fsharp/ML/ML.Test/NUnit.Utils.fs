module NN.Tests.NUnit.Utils
open NUnit.Framework
open NUnit.Framework.Constraints

type Range = Within of float array * float
let (+/-) a b = Within(a, b)

let equal x = 
    match box x with     
    | :? Range as r ->
        let (Within(x, within)) = r
        (new EqualConstraint(x)).Within(within)    
    | _ ->
        new EqualConstraint(x)

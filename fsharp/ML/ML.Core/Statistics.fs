module ML.Core.Statistics

open System

// Statistics monoids

type Distr = { TotalSq : float;  Total : float; Count : float}

let addDistr d1 d2 =
    { TotalSq = d1.TotalSq + d2.TotalSq  ; Total = d1.Total + d2.Total ; Count = d1.Count + d2.Count }

let (++) = addDistr

let distr n = { TotalSq = n * n ; Total = n ; Count = 1. }

let calcAvg d =
    if d.Count = 0. then 0. else float d.Total / float d.Count

//https://www.khanacademy.org/math/statistics-probability/displaying-describing-data/pop-variance-standard-deviation/v/statistics-alternate-variance-formulas

let calcVarS d =
    if d.Count <> 0. then 
        let avg = calcAvg d
        let cnt = d.Count - 1.
        d.TotalSq / cnt - 2. * avg * d.Total / cnt + avg * avg * d.Count / cnt
    else 
        0.

let calcVarP d =
    if d.Count <> 0. then 
        let avg = calcAvg d
        d.TotalSq / d.Count - avg * avg
    else 
        0.

let calcStdDevS = calcVarS >> Math.Sqrt

let calcStdDevP = calcVarP >> Math.Sqrt

let zero = { TotalSq = 0.;  Total = 0. ; Count = 0. }

// Avg
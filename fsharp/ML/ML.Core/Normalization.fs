module ML.Core.Normalization

open Statistics

let normalize model v = (v - model.Mu) / model.Std


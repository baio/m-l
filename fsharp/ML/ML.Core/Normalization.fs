module ML.Core.Normalization

type Batch = float seq seq

type NormModel = { Mu : float ; Std : float }

let normalize v model = (v - model.Mu) / model.Std


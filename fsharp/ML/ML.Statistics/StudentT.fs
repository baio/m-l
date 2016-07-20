module ML.Statistics.StudentT

open ML.Statistics.Gauss

let studentT (df: int) t =

  // for large integer df or double df
  // adapted from ACM algorithm 395
  // returns 2-tail p-value
  let muatble n = df // to sync with ACM parameter name
  let mutable a, b, y = 0., 0., 0. 
  let t2  = t * t
  y <- t2 / float df
  b <- y + 1.0
  if y > 1.0E-6 then y <- System.Math.Log(b)
  a <- float df - 0.5;
  b <- 48.0 * a * a;
  y <- a * y
  y <- (((((-0.4 * y - 3.3) * y - 24.0) * y - 85.5) /
            (0.8 * y * y + 100.0 + b) + y + 3.0) / b + 1.0) *
            System.Math.Sqrt(y)
  2.0 * (gauss <| -1. * y) // ACM algorithm 209

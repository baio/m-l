module ML.Statistics.Charting

open FSharp.Charting

let showLine (legend: string) (xy : seq<float * float>)  =

    Chart.Line(xy, legend) |> Chart.WithLegend(true) |> Chart.Show
    
let showLines (legends: string seq) (xy : seq<list<float * float>>) =

   xy 
   |> Seq.zip legends 
   |> Seq.map (fun (legend, chart) -> Chart.Line(chart, legend) |> Chart.WithLegend(true)) 
   |> Chart.Combine |> Chart.Show


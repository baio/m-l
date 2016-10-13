module ML.Core.Batch.Statistics

open ML.Core.Statistics
open ML.Core.Normalization

type Batch = float seq seq 

let batchOfList : List<List<float>> -> Batch = Seq.ofList >> Seq.map Seq.ofList

type ZipSeqBuilder() =

    member this.Apply f s = 
        Seq.map2 (fun x y -> x(y)) f s

    member this.Return x = 
        Seq.initInfinite (fun _ -> x)

    member this.Map f s =
        this.Apply (this.Return f) s
    
    member this.Lift2 f a b =         
        this.Apply (this.Map f a) b
    

let zipSeq = new ZipSeqBuilder()

let calcBtachNorm (calcNorm: Distr -> NormModel) (batch: Batch) =   
         
    batch |> Seq.fold (fun acc v ->
        zipSeq.Lift2 (++) (v |> Seq.map distr) acc 
    ) (Seq.initInfinite (fun _ -> distrZero()))

let calcBtachNormModel (calcNorm: Distr -> NormModel) (batch: Batch) =    
    
    batch |> calcBtachNorm calcNorm |> Seq.map calcNorm

let normalizeBatch (calcNorm: Distr -> NormModel) (batch: Batch) = 
    
    let normModel = calcBtachNormModel calcNorm batch 

    batch |> Seq.map (zipSeq.Lift2 normalize normModel)

let calcBtachNormModelP : Batch -> NormModel seq = calcBtachNormModel calcNormModelP 
let calcBtachNormModelS : Batch -> NormModel seq = calcBtachNormModel calcNormModelS
let normalizeBatchP : Batch -> Batch = normalizeBatch calcNormModelP
let normalizeBatchS : Batch -> Batch = normalizeBatch calcNormModelS
    

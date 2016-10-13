module ML.Core.Batch.Statistics

open ML.Core.Statistics
open ML.Core.Normalization

type Batch = float seq seq 

let batchOfList : List<List<float>> -> Batch = Seq.ofList >> Seq.map Seq.ofList

type BatchBuilder() =

    member this.Apply f s = 
        Seq.fold2 (fun acc x y -> x(y)) s f (this.Return(0.))

    member this.Return x = 
        Seq.initInfinite (fun _ -> Seq.initInfinite(fun _ -> x) )

    
type ZipSeqBuilder() =

    member this.Apply f s = 
        Seq.map2 (fun x y -> x(y)) f s

    member this.Return x = 
        Seq.initInfinite (fun _ -> x)

    member this.Map f s =
        this.Apply (this.Return f) s
    
    member this.Lift2 f a b =         
        this.Apply (this.Map f a) b

    member this.Bind(ev:seq<'T>, loop:('T -> seq<'U>)) : seq<'U> = 
      Seq.collect loop ev

    member this.For(ev:seq<'T>, loop:('T -> seq<'U>)) : seq<'U> = 
      this.Bind(ev, loop)
    
    member this.Yield(v:'T) : seq<'T> = seq [v]

    [<CustomOperation("zip",IsLikeZip=true)>]
    member this.Zip( outerSource:seq<'Outer>,  innerSource:seq<'Inner>, resultSelector:('Outer -> 'Inner -> 'Result)) : seq<'Result> =
        Seq.map2 resultSelector outerSource innerSource

let zipSeq = new ZipSeqBuilder()

let (<!>) a b = zipSeq.Map
let (<*>) a b = zipSeq.Apply

let calcBtachNorm (calcNorm: Distr -> NormModel) (batch: Batch) =   
         
    batch |> Seq.fold (fun acc v ->
        zipSeq.Lift2 (++) (v |> Seq.map distr) acc 
        //v |> Seq.map distr |> Seq.zip acc |> Seq.map (fun (a, b) -> a ++ b)
        // (++) <!> (v |> Seq.map distr) <*> acc
        //zipSeq.Lift2 (++) (v |> Seq.map distr) acc 
        (*
        zipSeq { 
            for x in acc do 
            zip y in (v |> Seq.map distr)
            yield x ++ y
        }
        *)
    ) (Seq.initInfinite (fun _ -> distrZero()))

let calcBtachNormModel (calcNorm: Distr -> NormModel) (batch: Batch) =    
    batch |> calcBtachNorm calcNorm |> Seq.map calcNorm

let normalizeBatch (calcNorm: Distr -> NormModel) (batch: Batch) = 
    
    let normModel = calcBtachNormModel calcNorm batch 

    batch |> Seq.map (zipSeq.Lift2 normalize normModel)
    (*
    (fun x ->
        zipSeq.Lift2 normalize normModel x
        //Seq.mapi (fun i e -> normalize ( normModel |> (Seq.item i) ) e)
     )
     *)

let calcBtachNormModelP : Batch -> NormModel seq = calcBtachNormModel calcNormModelP 
let calcBtachNormModelS : Batch -> NormModel seq = calcBtachNormModel calcNormModelS
let normalizeBatchP : Batch -> Batch = normalizeBatch calcNormModelP
let normalizeBatchS : Batch -> Batch = normalizeBatch calcNormModelS
    

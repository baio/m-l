#r @"..\libs\TypeShape.dll"

open TypeShape

type Rec<'a> = {
    Theta : float
    Params : 'a
}

    

let foo<'a> i : Rec<'a> =    

    match i with
    | 0 ->                           
        let a = box 0. :?> 'a
        { Theta = 0.; Params = a }
    | _ ->                        
        let a : 'a = box 1 :?> 'a   
        { Theta = 1.; Params = a }


let foo2 i : Rec<'a> =    

    match i with
    | 0 ->           
        let ts = TypeShape.Create (typeof<Rec<int>>)      
        match ts with
        | Shape.FSharpRecord2 s ->
            s.Accept {
                new IFSharpRecord2Visitor<obj> with
                    member __.Visit<'Record,'Field1,'Field2> (s : IShapeFSharpRecord<'Record,'Field1,'Field2>) =
                       { Theta = 0.; Params = 1 } :?> _
            }
        | _ -> failwith "not defined"
    | _ ->                        
        let a : 'a = box 1 :?> 'a   
        { Theta = 1.; Params = a }


let defoo<'a> f : 'a =    

    match box f with
    | :? Rec<int> as x ->                           
        box x.Params :?> 'a
    | :? Rec<float> as x ->                           
        box x.Params :?> 'a
    | _ -> failwith "unknown"

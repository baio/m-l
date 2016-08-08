

type Rec<'a> = {
    Theta : float
    Params : 'a
}
    

let foo<'a> i : Rec<'a> =    

    match i with
    | 0 ->                           
        let a : 'a = box 0. :?> 'a
        { Theta = 0.; Params = a }
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

module ML.DGD.Types

open MathNet.Numerics.LinearAlgebra

open ML.GD.GLM
open ML.GD.GradientDescent
open ML.GD.GD
open SamplesStorage
     
type BatchSamples = 
    //Samples splited in GDD Batch Coordintor and provided for each batch
    | BatchSamples of float Matrix *  float Vector
    //Batch indexes generated in coordinator, GDD Batch should read them from storage by index
    | BatchSamplesIndexes of SamplesStorage * int list
    //GDD Batch should randomly generate indexes of size in and then read samples from storage
    | BatchSamplesStochastic of SamplesStorage * int
    // Request samples from samples server
    | BatchSamplesServer // TODO

type DGDParams = {    
    Model: GLMModel
    HyperParams: GradientDescentHyperParams
    EpochNumber: int
    //Samples storage
    SamplesStorage: SamplesStorage
    //Distributed batch size
    DistributedBatchSize: int
    //How GDBatch get samples
    BatchSamples: GDDBatchSamples 
} and GDDBatchSamples = 
    | BatchSamplesProvidedByCoordinator
    | BatchSamplesIndexesProvidedByCoordinator
    | BatchSamplesStochastic
    | BatchSamplesServer

type BatchesMessage =
    | BatchesStart of DGDParams    
    | BatchCompleted of ModelTrainResult
    | BatchesCompleted of ModelTrainResult
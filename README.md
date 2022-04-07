Code for :

Attention Enhanced Time Series Classification: An approach for implantation outcome prediction in time-lapse human embryo image sequences

python 3.6
torch 1.2.0
####################################

UEA Benchmark for "Global Temporal Attention block" and "Temporal Pseudo-Gaussian augmented Self-attention"
UEA archive should be extracted in "Multivariate_ts" folder, sktime ts format.


###########################
Run ueatrain.py with the following the specified arguments for different models.

###########################

    FCN               :        mode=1,   ffh=16
    FCN + GTA         :        mode=301, ffh=16
    FCN + TPS         :        mode=400, adaD=3
    
    ResNet            :        mode=2,   ffh=16
    ResNet + GTA      :        mode=122, ffh=16
    ResNet + TPS      :        mode=400, adaD=4
    
    Inception32       :        mode=1400, hids=32 ,   ffh=16
    Inception32 + GTA :        mode=1402, hids=32 ,   ffh=16
    Inception32 + TPS :        mode=1408, hids=32
    
    
    Inception64       :        mode=1400, hids=64 ,   ffh=16
    Inception64 + GTA :        mode=1402, hids=64 ,   ffh=16
    Inception64 + TPS :        mode=1408, hids=64 
    
    
    E+TPS             :        mode=400, adad=1
    E+SA              :        mode=400, adad=1, LrMo=3
    
    TPS---------------
    No PE             : pos_enc=0
    Learnable PE      : pos_enc=2
    Fixed Function PE : pos_enc=1
    non TPS-----------
    pos_enc=0
    -------------
############################

set no between 0 t0 29 for specific dataset.

    ArticularyWordRecognition
    AtrialFibrillation
    BasicMotions
    CharacterTrajectories
    Cricket
    DuckDuckGeese
    EigenWorms
    Epilepsy
    ERing
    EthanolConcentration
    FaceDetection
    FingerMovements
    HandMovementDirection
    Handwriting
    Heartbeat
    InsectWingbeat
    JapaneseVowels
    Libras
    LSST
    MotorImagery
    NATOPS
    PEMS-SF
    PenDigits
    Phoneme
    RacketSports
    SelfRegulationSCP1
    SelfRegulationSCP2
    SpokenArabicDigits
    StandWalkJump
    UWaveGestureLibrary

##############################

set --JobId to different numbers every run and average the results.

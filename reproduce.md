# Partitioning the dataset

## BDDATA

1. use script `make_training_test_folds.py` to make the *folded* data. Remarks:
    - parameters are set in the script itself (despite the latter needing to read `parameters.json`
    - `dismiss_single_measurements` controls wheter subjects with a single measurement must be dropped
    - set `data_directory` to point to the directory where the original data reside, and...
    - ...specify the names of the individual files for the biomarkers `CA125_healthy_subjects_file`, `CA125_ill_subjects_file`, ...
    - around the middle of the script there is commented/uncommented code to setup how the data is to be partitioned
        - `StratifiedShuffleSplit` will sample the data allowing as many folds as desired, which entails that a single sample (subject) may show up in several *training* and *test* folds
        - `StratifiedKFold` is *classic* cross-validation
1. That will yield a collection of *fold_*? directories, each one with a different training/test partition **for each biomarker**

## UKCTOCKS

Soon...maybe...

# Running the simulation

1. Run *bash* script `evaluate_on_folds.sh` passing as argument the **full path** to the directory in which folds *fold_*? were created. Remarks:
    - a `parameters.json` file must exist in the directory from which `evaluate_on_folds.sh` is run, and an link to it will be created inside every *fold_*? directory. A workflow example:
    ```
    export SIM_DIR="simulations/5_folds"
    export SCRIPTS_PATH="/home/manu/rnns"
    export FOLDED_DATA_PATH="/home/manu/rnns/folded_data"
    
    mkdir $SIM_DIR
    pushd $SIM_DIR
    
    # a symbolic link to the relevant parameters file (assuming more than one) 
    ln -s ../TimedCA125_parameters.json parameters.json
    
    # simulation is run
    $SCRIPTS_PATH/evaluate_on_folds.sh $FOLDED_DATA_PATH/5_folds
    
    popd
	```

# Collecting results

1. 

# Troubleshooting

## Mac

It seems runtime errors may occur depending on the version of some of the libraries. The libraries in the `rnns` environment built using
```
conda create --name rnns python=3.6 tensorflow=1.1 keras=2.0 colorama pandas scikit-learn matplotlib
```
are known to work out fine.
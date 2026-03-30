# Cam_CAN_MEG_Resting_State_Pipeline

This repository contains a pipeline to clean and create source localized data for the resting state Cam-CAN MEG dataset. This pipeline was created by Dr. Simon Dobri and developed by Dr. Jack Solomon for his 2026 article, titled "Interactions between age and sex in multiscale entropy and spectral power changes across the lifespan".

## Repository Outline

This repository contains code that:
- Uses freesurfer to create parcellated structural MRI data
- Extracts window of resting state data and cleans it using a bandpass and notch filter as well as either SSP or ICA motion correction.
- Projects data from sensor to source space using either a volumetric
  - Optionally parcellates the source level data

## Dependencies

The dependencies for this analysis are as follows. 

- freesurfer (latest)

- mne python (latest)
  - If using the fir preinstalled mne wheel the additional dependencies are:
    - nibabel (latest)
    - sklearn (latest)
    - python-picard (latest)

The scripts were run in python 3.11.4 and use freesurfer 7.4.1 for publication. They may work with other versions of python but are not guaranteed to function correctly.

## Raw data

Access to the raw data, the [Cam-CAN](https://cam-can.mrc-cbu.cam.ac.uk/dataset/) repository, has controlled access through Cambridge University. One can ask for ask for data access [here](https://opendata.mrc-cbu.cam.ac.uk/projects/camcan/request/).

After running the git clone command, `cd` into the cloned directory. All commands should be executed from the parent directory of the git repository.

To replicate the results of solomon et al. (2026). You will need:
  - The raw resting state MEG data for each participant from release 005 with only a maxfilter applied (i.e. without movement compensation)
    - the subject directories should be places in a new directory `./Solomon_2026_MEG_pipeline/_Data/meg/meg_restingstate/` 
  - The emptyroom recordings for each participant
    - the subject directories should be places in a new directory `./Solomon_2026_MEG_pipeline/_Data/meg/meg_emptyroom/` 
  - the MEG registration files for each participant
    - the subject directories should be places in a new directory `./Solomon_2026_MEG_pipeline/_Data/meg/camcan_coreg/` 
    - the transformation files were provided by [Bardouille et al. 2019](https://www.sciencedirect.com/science/article/pii/S1053811919301612?via%3Dihub)
  - the raw structural MRI files for each participant
    - the subject directories should be places in a new directory `./Solomon_2026_MEG_pipeline/_Data/mri/`

## Code Usage

The experiment's code is formatted for use on the [Digital Research Alliance of Canada](https://www.alliancecan.ca/en) Fir cluster and will need to be adapted for local processing.

## Running the FreeSurfer pipeline

To run the FreeSurfer pipeline use the following code blocks:

```
sbatch ./freesurfer_scripts/freesurfer_script.sh
```

Once this job is completed then run:

```
sbatch ./freesurfer_scripts/freesurfer_run_sub-CC320022.sh
sbatch ./freesurfer_scripts/freesurfer_run_sub-CC721704.sh
```

## Running the MEG pipeline

Once the data is loaded, the pipeline can be run using the batch script `./batch_scripts/submit_beamformer_subjects.sh`.

To submit the job to the compute clusters, ensure that the working directory is the git repository's parent directory and use the following code:

```
./batch_scripts/submit_beamformer_subjects.sh meg_environment.sif ./tvb-ccmeg/pipeline_rest_beamformer.py ./batch_scripts/subject_list.txt
```

This will create the processed MEG data files in `./_Data/processed_meg/`, which can be used as the inputs to the [camcan_mse_analysis](https://github.com/McIntosh-Lab/camcan_mse_analysis) for repoduction of the results of Solomon et al. 2026.

# Code used to analyze effect of P.aeruginosa colonies on growth of S. aureus WT and S aureus gltT mutant colonies

## Reproduce figures

Run the `6_plot_colony_growth` notebook

## Description full pipeline

Create conda environment from environment file:

`conda env create -f environment.yml`

### 0. Data preparation

- `0_create_metadata` notebook stores metadata for each experiment in csv pandas data frame
- `1_register_export_data` notebook trims movies, registers to compensate for pad movement and write registered movies to disk, uses code in `registration.py`

### 1. Segmentation in Ilastik

- `2_create_training_dataset` notebook exports subset of data into hfd5 file to import into Ilastik for classifier training

Train Ilastik classifier and export probability maps

### 2. Post-process and track in Python

- `3A_check_segmentation_streamlined` notebook to check Ilastik segmentation and set post-processing settings, uses code in `process_colonies.py`
- `3B_check_segmentation` notebook same as above but explaining processing step by step (for illustration purposes only)
- `4_batch_process_segmentation` notebook apply postprocessing (check in 3A) to all data, uses code in `process_colonies.py`. Output stored in `all_data.csv`.

### 3. Analyze data

- `5_filter_data` notebook applies filtering on colony tracks. Reads in `all_data.csv` and stores output in `all_data_filtered.csv`.
- `6_plot_colony_growth` notebook plots results. Reads in `all_data_filtered.csv`.
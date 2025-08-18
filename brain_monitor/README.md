# brain_monitor
Repository for detecting datadrift. We use kolmogorov-smirnoff tests to detect a significant difference between the feature distribution of each eeg channel.

## src 
- drift_detection.py: Detects a datadrift between the feature distribution for each state of a new person and the data in the storage. Therefore the signatur and the path to the 3 csv files of the new person is needed. The kolmogorov smirnoff test is used for each channel for each dataframe of the given state. The results are saved in a new dataframe for each state containing the following columns:
    - Feature: Corresponds to the EEG channel
    - Pval: P value of the kolmogorov smirnoff test
    - DF Hash: Hash value of the dataframe which is used for the test
    - Date: Date of the experiment which represents the other dataframe
    - Drift Detected: True if the Pval is less than the alpha value. The alpha value is corrected with the [Bonferroni Correction](https://de.wikipedia.org/wiki/Bonferroni-Korrektur)

For each of those dataframe a sliding window with size 5 is used to go over the pvalues (sorted by time) to capture the amounts of 5 consecutive datadrifts for each state. An alarm can be set if the count is greater than 1.

# Quickstart
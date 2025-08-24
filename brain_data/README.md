# Brain Data
This repo builds genetic docker image to track data versions with [DVC](https://dvc.org/) in following Repos:

* [brain_data_raw](https://code.fbi.h-da.de/mlops-brain/brain_data_raw)
* [brain_data_processed](https://code.fbi.h-da.de/mlops-brain/brain_data_processed)
* [brain_feature_store](https://code.fbi.h-da.de/mlops-brain/brain_feature_store)

## Docker
Required environment variables (see [linage.env.sample](./linage.env.sample)):
```
BRAIN_REPO_TOKEN=
DVCFOLDER= 
```
Required mount points:
```
./data:/app/collect_point # the external ./data folder receives files to be tracked
.aws:/root/.aws # for connection with s3 bucket
```
Once start, the container:
* pulls repo `brain_{DVCFOLDER}` with `BRAIN_REPO_TOKEN`.
* If there are new files under `/app/collect_point`, it: 
  * moves all new files to the folder `/app/brain_{DVCFOLDER}/{DVCFOLDER}`
  * tracks files with DVC, push changes to connected s3 bucket.
  * commits and pushes git message
* cleans folders and close

## Quickstart
* Copy AWS credentials under this folder. (ask `helmut.liu@stud.h-da.de` for credentials)
* Define required environment variables in `linage.env` like it in `linage.env.sample`
* Run
```
docker-compose up --build
```
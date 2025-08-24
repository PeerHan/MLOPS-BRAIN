# ETL
This folder initially contained the whole etl pipeline which now was largely moved to airflow.
Remaining is the brain receiver container and the corresponding brain_data instance.
Brain Receiver serves the api which brain_collector is sending the data from the EEG Headset to.
It then does three things:
1. it stores the data to the data directory
2. it sends the data to infer.mlopsbrain.world for model inference
3. it sends the data to our drift detection  container which monitors for data drift and sends out a notification to request a new data collection campagne.

In order to host the service you need setup a password and user for the receiver api inside `receiver.env`, store the aws credentials inside the `.aws`folder and the needed variables inside `linage_raw.env`.

When finished the previouse steps simply run `docker compose up -d`
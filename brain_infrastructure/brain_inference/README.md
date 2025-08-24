# Brain Inference
This containe, when started, looks for the currently best model pairing with the currently deployed transform pipeline and serves the inference api on the incoming data.

To run the service you need to enter the credentials for the mlfow server inside the `.env` file.
Afterwards simply run `docker-compose up -d`.
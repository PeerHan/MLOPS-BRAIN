# brain_infrastructure
## Server Setup
### Initial Setup
#### SSH Configuration
```bash
sudo nano /etc/ssh/sshd_config
```
Change the following lines to the following values:
```bash
PasswordAuthentication no
Banner none
PermitRootLogin no 
```
#### Docker Install
Install Docker as described [here](https://docs.docker.com/engine/install/ubuntu/)

After Installation add useres as needed to the docker group so no sudo is needed for docker commands:
```bash

### User Management
```bash
sudo adduser --disabled-password --gecos "" user1
sudo mkdir -p /home/user1/.ssh
sudo chown sttiseess:sttiseess /home/sttiseess/.ssh
sudo chmod 700 /home/sttiseess/.ssh
echo "<PUBLIC_KEY>" | sudo tee /home/sttiseess/.ssh/authorized_keys
sudo chown sttiseess:sttiseess /home/sttiseess/.ssh/authorized_keys
sudo chmod 600 /home/sttiseess/.ssh/authorized_keys
sudo usermod -aG docker user1
sudo usermod -aG docker_services user1
```
Create group to manage permissions for docker services directory:
```bash
sudo groupadd docker_services
sudo usermod -aG docker_services sttiseess
sudo chgrp -R docker_services /services
sudo chmod -R 775 /services
sudo chmod g+s /services
```

### Architecture
```mermaid
graph TD
    title(MLOps-BRAIN - Birthdate Retrieval from Analysis of Inferred Neurodata)

    subgraph Laptop
        A[Brain Collector]
    end

    subgraph Server
        B[Brain Receiver]
        C[Brain Data]
        D[Brain Transform]
        E[Brain Model]
        F[MLflow Tracking Server]
        G[MLflow Model Registry]
        H[Optuna HPO]
        I[Deployed API]
        DVCRepoRaw((DVC Repo - Raw Data))
        DVCRepoProcessed((DVC Repo - Processed Data))
        DVCRepoViz((DVC Repo - Visualization Data))
    end

    %% Connections
    A -->|Sends data via API| B
    B -->|Stores data into folder| C
    C -->|Copies data to Raw DVC repo and commits| DVCRepoRaw
    C -->|Copies data to ETL Start folder| ETLStart[ETL Start Folder]
    %% C -->|Processes new data| D
    D -->|Stores processed data| Processed[ETL Processed Folder]
    D -->|Stores visualizations| Viz[ETL Viz Folder]
    Processed -->|Watched by Brain Data| DVCRepoProcessed
    Viz -->|Watched by Brain Data| DVCRepoViz
    DVCRepoViz -->|Watches visualization data| E

    %% Model pipeline additions
    E -->|Logs metrics| F
    E -->|Runs hyperparameter optimization| H
    H -->|Sends optimized model| G
    F -->|Stores trained models| G
    G -->|Deploys best model| I
    I -->|Provides access via API| API[External API]

    %% Watching triggers
    D -->|Watches folder for new data| ETLStart
    D -->|Watches ETL Start folder for new data| Processed

```
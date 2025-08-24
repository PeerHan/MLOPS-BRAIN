# brain_dags
## DAGs
### [run_brain_transform.py](run_brain_transform.py)
Dieser DAG koordiniert die Datenvorverarbeitungspipeline für die Brain Daten. Die Pipeline besteht aus folgenden Schritten:

1. **check_dag_active**:
   - Überprüft, ob eine andere Instanz dieses DAGs bereits läuft und überspringt die Ausführung, falls dies der Fall ist.

2. **run_brain_transform**: 
   - Führt die Haupttransformation der Brain-Daten mittles dem [`brain_transform`](https://code.fbi.h-da.de/mlops-brain/brain_transform/container_registry/3006) Container
   - Liest Rohdaten aus S3
   - Verarbeitet die Daten und speichert sie in zwei Ausgabeordnern die persistent auf die Festplatte:
     - `/app/processed`: Für prozessierte Daten
     - `/app/viz`: Für visualisierungsfertige Daten

3. **check_run_docker_output**:
   - Überprüft die Logs des Docker-Containers auf die Phrase "New data was transformed"
   - Überspringt nachfolgende Tasks, wenn diese Phrase nicht gefunden wird (keine neuen Daten)

4. **brain_load_processed** und **brain_load_viz**:
   - Diese Tasks werden nur ausgeführt, wenn IS_DEVELOPMENT auf False gesetzt ist
   - Parallel laufende Tasks nach der Transformation um die prozessierten und visualisierungsfertigen Daten in einem DVC Repo mit AWS S3 Backend zu speichern mittels [`brain_data`](https://code.fbi.h-da.de/mlops-brain/brain_data/container_registry/2929) Container
   - Tracken die verarbeiteten Dateien in einem DVC Repository
   - Ermöglichen Nachverfolgung und Monitoring des Datenflusses

5. **trigger_model_dag**:
   - Löst den `model_trigger_dag` aus, um das Modelltraining zu starten
   - Wartet auf den Abschluss des ausgelösten DAGs

Der DAG läuft stündlich und verwendet Docker-Container für die Ausführung aller Tasks.
Im Entwicklungsmodus (IS_DEVELOPMENT = True) werden die Datenlade-Tasks übersprungen.
Alle benötigten Umgebungsvariablen werden aus entsprechenden .env Dateien geladen.

### [run_brain_model_check.py](run_brain_model_check.py)
Dieser DAG überprüft, ob ein Modelltraining erforderlich ist, und startet es gegebenenfalls. Die Pipeline besteht aus folgenden Schritten:

1. **check_dag_active**:
   - Überprüft, ob eine andere Instanz dieses DAGs bereits läuft und überspringt die Ausführung, falls dies der Fall ist.

2. **run_model_trigger**:
   - Führt den [`brain_model/trigger`](https://code.fbi.h-da.de/mlops-brain/brain_model) Container aus, um zu überprüfen, ob ein Modelltraining erforderlich ist.

3. **check_mismatch**:
   - Überprüft die Logs des `run_model_trigger` Containers
   - Bei DEV_SKIP_TRIGGER = True wird direkt das Modelltraining gestartet
   - Bei DEV_SKIP_MODEL = True wird der DAG beendet
   - Im normalen Betrieb: Prüft auf die Phrase "File counts differ between the dataset commit and master"
   - Verzweigt basierend auf dem Ergebnis zum Modelltraining oder zum Ende

4. **start_model_training**:
   - Startet das Modelltraining mittels dem [`brain_model/app`](https://code.fbi.h-da.de/mlops-brain/brain_model) Container
   - Wird nur ausgeführt, wenn ein Mismatch festgestellt wurde oder DEV_SKIP_TRIGGER = True

5. **end_dag**:
   - Dummy-Task, der das Ende des DAGs markiert, wenn kein Training erforderlich ist

Der DAG wird durch den run_brain_transform DAG ausgelöst und verwendet Docker-Container für alle Tasks.
Entwicklungsflags (DEV_SKIP_TRIGGER, DEV_SKIP_MODEL) ermöglichen verschiedene Testszenarien.
Alle benötigten Umgebungsvariablen werden aus entsprechenden .env Dateien geladen.

#### Zusätzliche Trigger
Zusätzlich kann die Modell Training Pipeline per Airflow API gestartet werden:
```bash
curl -X POST \                                          
     -u "user:pwd" \
     -H "Content-Type: application/json" \
     -d '{}' \
 https://airflow.mlopsbrain.world/api/v1/dags/model_trigger_dag/dagRuns
```

## CI/CD
### Generelle Überlegungen
Airflow wird nur genutzt um Docker Container zu starten.
Dies ermöglicht ein schlankes airflow Deployment.
Zusätzlich haben wir uns dazu entschieden sensible Daten die aus .env Dateien in die Container gezogen werden manuell auf dem Server zu platzieren, um möglichst wenig sensible Daten unverschlüsselt auf Gitlab speichern zu müssen, und da diese sich normalerweise nicht ändern.
Damit die DAGs hingegen schnellstmöglich aktualisiert werden können werden diese aktiv von git auf den Projekt-Server gepushed.
Grundsätzlich wäre es noch wünschenswert den Service User so einzurichten, z.B: mit einem SFTP-Chroot Ansatz, so dass dieser nur die benötigten Rechte hat, um die DAGs zu kopieren und nicht mehr ausführen, lesen oder schreiben kann, was aber aus Zeitgründen nicht umgesetzt wurde.

### Gitlab CI/CD
Damit Gitlab Actions die DAGS deployen kann, also auf den Airflow Server kopieren kann, muss ein Service Account erstellt werden und die SSH Keys müssen hinterlegt werden.

1. SSH Keys ohne passphrase erzeugen und im aktuellen Verzeichnis speichern. 
    ```bash 
    ssh-keygen -t ed25519 -C "brain_ci" -f ./depl_ed25519 -N ""
    ```
2. Auf dem Airflow Server den User anlegen
    ```bash
    # Create the user
    sudo useradd -m -s /bin/bash brain_ci

    # Create .ssh directory and set permissions
    sudo mkdir -p /home/brain_ci/.ssh
    sudo chmod 700 /home/brain_ci/.ssh

    # Create authorized_keys file and add your public key
    sudo touch /home/brain_ci/.ssh/authorized_keys
    sudo chmod 600 /home/brain_ci/.ssh/authorized_keys

    # Add the public key (this corresponds to your CI_SSH_PRI private key)
    sudo sh -c 'echo "YOUR_PUBLIC_KEY_HERE" >> /home/brain_ci/.ssh/authorized_keys'

    # Set correct ownership
    sudo chown -R brain_ci:brain_ci /home/brain_ci/.ssh
    ```
3. Erlaubniss für den User `brain_ci` auf dem Airflow Server erteilen um mit sudo privilegen den dags zu kopieren. Dafür die folgende Zeile in der Datei `/etc/sudoers` anhängen mittles dem `visudo` Befehl.
    ```bash
    brain_ci ALL=(ALL) NOPASSWD: /usr/bin/cp -a /home/brain_ci/temp_dags/* /services/brain_infrastructure/airflow2/dags/
    ```



### Einschränkungen
Mit dem CI/CD können Dateien nur aktualisiert oder hinzugefügt werden, nicht aber gelöscht werden, was in der Praxis kein Problem darstellt und daher nicht weiter beachtet wurde.
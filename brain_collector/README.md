# Brain Collector
Schnittstelle zur Extraktion von Zahlen aus dem menschlichen Gehirn mittels Hirnwellensensor.

## Projektstruktur
- `src/`: Quellcode-Verzeichnis
  - `receive_data.py`: Programm zum Empfangen der EEG-Daten
- `app.py`: Hauptanwendung mit Streamlit-Interface
- `environment.yml`: Conda-Umgebungskonfiguration

## Schnellstart

### Installation
1. Conda-Umgebung erstellen:
```bash
conda env create -f environment.yml
```

2. Umgebung aktivieren:
```bash
conda activate brain
```

### Umgebungsvariablen
Erstellen Sie eine `.env`-Datei mit folgenden Einstellungen:
```
API_URL=http://127.0.0.1:8000/upload/
API_USERNAME=user
API_PASSWORD=password
TIME_INTERVAL=0.1
START_YEAR=1999
END_YEAR=2007
```

### Verwendung
1. Neurosity Crown einschalten
2. Sicherstellen, dass Laptop und Crown im selben WLAN sind
3. Web App mit folgendem Befehl starten:
    ```bash
    streamlit run app.py
    ```
4. Im Browser öffnet sich die Benutzeroberfläche automatisch. Mit einem Klick **Start Animation and Data Capture** wird das Experiment gestartet. 

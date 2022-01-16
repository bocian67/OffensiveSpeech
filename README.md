# Installation
- In dieses Repo die trainingsdaten ("train"-Ordner) kopieren
- Im terminal (Es wird angenommen dass Python 3 installiert ist):
    ```bash
    pip install -r requirements.txt
    python -m spacy download de_core_news_lg
    ```

# Benutzung
Training und Testing mittels der bereitgestellten Daten:
```bash
python main.py
```

Cross-Validation:
```bash 
python Cross_validation.py
```

Gridsearch (Parametersuche)
```bash
python gridsearch.py
```
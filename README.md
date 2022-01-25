# Installation
- Im terminal (Es wird angenommen dass Python 3 installiert ist):
    ```bash
    pip3 install -r requirements.txt
    python3 -m spacy download de_core_news_lg
    ```

# Benutzung
Training und Testing mittels der bereitgestellten Daten:
```bash
python3 main.py
```

Cross-Validation:
```bash 
python3 Cross_validation.py
```

Gridsearch (Parametersuche)
```bash
python3 gridsearch.py
```

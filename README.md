# processor_generator

## Interface Tkinter pour visualiser le développement de l'IA

Un visualiseur desktop (`tkinter`) est disponible dans `tkinter_ai_dashboard.py`.

### Fonctionnalités
- Lancer/arrêter un script d'entraînement IA (ex: `cpu_circuit_builder.py`).
- Afficher les logs en direct (`stdout` + suivi du `run.log`).
- Lire `out_design/training_log.csv` et afficher les dernières itérations.
- Mettre en évidence le meilleur score détecté.

### Lancer l'interface
```bash
python tkinter_ai_dashboard.py
```

Ensuite:
1. Sélectionnez le script IA à exécuter.
2. Sélectionnez le dossier d'output (par défaut `out_design`).
3. Cliquez **Lancer**.

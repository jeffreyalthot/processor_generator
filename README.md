# processor_generator

## Interface Tkinter pour visualiser le développement de l'IA

## Nouveau point d'entrée unifié (`main.py`)

Pour utiliser les deux scripts ensemble *en live* (dashboard + lancement automatique de l'IA), lancez:

```bash
python main.py --mode live
```

Modes disponibles:
- `--mode live` : ouvre le dashboard **et** démarre automatiquement `cpu_circuit_builder.py`.
- `--mode dashboard` : ouvre seulement l'interface Tkinter (démarrage manuel via le bouton).
- `--mode trainer` : exécute seulement l'entraînement en console.

Options utiles:
- `--script cpu_circuit_builder.py`
- `--out-dir out_design`
- `--design-budget-sec 300`
- `--eval-cycles 1000`

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

# Projet Aurore Boréale (Avatare)

**Le fichier predict.py est configurée sur un modèle mac donc "mps".
Si vous voulez utiliser le modèle passer de "mps" à "gpu"**

## Transformer
**Avant d’exécuter les commandes veuillez vous positionner dans le dossier transform.**

### Train
La commande train permet de lancer l’entrainement du model.
```shell
python predict.py train
```

### Retrain
Continue l’entraînement à partir d’un modèle déjà existant.
```shell
python predict.py retrain
```

### Validate
La commande validate permet d’évaluer les performances du modèle en utilisant les 10 % les plus récents des données, 
sur lesquels il n’a pas été entraîné. Une image sera générée, montrant les données réelles ainsi que les prédictions du 
modèle.
```shell
python predict.py validate
```

### Prédiction sur realtime, mag1H et mag-tempête1H
- Permet de faire une prédiction sur les données en temps réels.
    ```
    python predict.py forecast --hours 1 --realtime
    ```
- Permet de faire une prédiction sur les données mag1H combiné à plasma-storm
    ```shell
    python predict.py forecast-files --mag ../../data/dataBizarre/mag-1h.json --plasma ../../data/dataBizarre/plasma-storm-1h.json --hours 1 --label "mag-1h + plasma-storm-1h"
    ```
- Permet de faire une prédiction sur les données mag-strom1H combiné à plasma-strom
    ```shell
    python predict.py forecast-files --mag ../../data/dataBizarre/mag-storm-1h.json --plasma ../../data/dataBizarre/plasma-storm-1h.json --hours 1 --label "mag-storm-1h + plasma-storm-1h"
    ```

## Différentes branches

- test/ddp : essaie pour mettre la distribution en place
- neural_network : mise en place des intégrales
- knn : test avec le modèle KNN
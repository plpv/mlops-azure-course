import json
import numpy as np
import os
import tensorflow as tf

from azureml.core.model import Model


def init():
    global tf_model
    model_root = os.getenv("AZUREML_MODEL_DIR")
    # the name of the folder in which to look for tensorflow model files
    tf_model_folder = "model"

    tf_model = tf.saved_model.load(os.path.join(model_root, tf_model_folder))

def run(mini_batch):
    results = []
    
    # Parcourir chaque fichier dans le mini-batch
    for file_path in mini_batch:
        
        # Lire le contenu du fichier
        with open(file_path, 'r') as f:
            file_contents = f.read()
        
        # Convertir les données au format attendu
        data = np.array(json.loads(file_contents)["data"], dtype=np.float32)
        
        # Faire la prédiction
        out = tf_model(data)
        y_hat = np.argmax(out, axis=1)
        
        # Ajouter les résultats
        results.append(y_hat.tolist())
    
    return results
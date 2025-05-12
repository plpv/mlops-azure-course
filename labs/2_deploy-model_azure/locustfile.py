# import boto3
# from locust import HttpUser, between, task
# import json 
# import numpy as np
# from PIL import Image
# import os 
# PROJECT_ID = "projet-ia-448520"  
# LOCATION = "us-central1"  
# ENDPOINT_ID = "5825974565515296768"

# def load_data():
#     IMAGE_DIRECTORY = "cifar_test_images"

#     image_files = [file for file in os.listdir(IMAGE_DIRECTORY) if file.endswith(".jpg")]
#     image_data = [
#         np.asarray(Image.open(os.path.join(IMAGE_DIRECTORY, file))) for file in image_files
#     ]
#     x_test = [(image / 255.0).astype(np.float32).tolist() for image in image_data]
#     y_test = [int(file.split("_")[1]) for file in image_files]
#     return x_test, y_test

# x_test, y_test = load_data()
# class APIUser(HttpUser):
#     wait_time = between(1, 3)  

#     def on_start(self):
#         from google.auth import default
#         from google.auth.transport.requests import Request

#         credentials, _ = default()
#         credentials.refresh(Request())
#         self.auth_token = credentials.token
#         #return super().on_start()

#     @task
#     def call_endpoint(self):
#         single_image = x_test[1]
        
        
#         payload = {'instances': [
#             json.loads(json.dumps(single_image))
#             ]
#             }
        
#         header = {
#             "Authorization": f"Bearer {self.auth_token}",
#             "Content-Type": "application/json"
#         }
#         response = self.client.post(f"/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict", json=payload, headers = header)
       
#         if response.status_code == 200:
#             print(response.json())
#         else:
#             print(f"Erreur {response.status_code}: {response.text}")

from locust import HttpUser, between, task
from locust import events
import time
import json
import numpy as np
from PIL import Image
import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

# Configuration Azure
SUBSCRIPTION_ID = "_____"
RESOURCE_GROUP = "_____"
WORKSPACE_NAME = "_____"
ENDPOINT_NAME = "_____" # endpoint de temps réel
SCORING_URI = "_____"
API_KEY = "_____"

def load_data():
    # Charger les données depuis le fichier JSON
    with open(f"{os.getcwd()}\\request\\sample-request.json", "r") as f:
        json_data = json.load(f)
    
    # Extraire les données d'image (une liste de 784 valeurs)
    # x_test = json_data["data"]
    
    return json_data

data = load_data()

class APIUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Authentification Azure
        credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME,
        )

    @task
    def call_endpoint(self):
        # Début du suivi de la requête
        start_time = time.time()
        
        try:
            # Appel du endpoint
            response = self.ml_client.online_endpoints.invoke(
                endpoint_name=ENDPOINT_NAME,
                request_file=f"{os.getcwd()}\\request\\sample-request.json"
            )
            
            # Calcul du temps de réponse
            response_time = (time.time() - start_time) * 1000  # en millisecondes
            
            # Enregistrement de la requête réussie
            events.request.fire(
                request_type="POST",
                name="AzureML Endpoint",
                response_time=response_time,
                response_length=len(str(response)),
                exception=None,
                context=self.context()
            )
            
            print("Response:", response)
            
        except Exception as e:
            # Enregistrement de la requête échouée
            events.request.fire(
                request_type="POST",
                name="AzureML Endpoint",
                response_time=0,
                response_length=0,
                exception=e,
                context=self.context()
            )
            print(f"Error: {str(e)}")
# End-to-end-Machine-Learning-Project-with-MLflow
# Rent_Prediction_Using_ML
A rent prediction webapp, based on Delhi's dataset. Includes complete CI/CD pipeline along with model tracking via MlFlow and AWS deployment ! 
The regression model is built using Pytorch.
<table>
	<tr>
 		<td><img src="https://github.com/user-attachments/assets/784d746c-5f33-4d80-9a7a-05d5fda7e155"></td>
   		<td><img src="https://github.com/user-attachments/assets/a3a89eea-f1c0-4b7e-9ba1-a10d5a591770"></td>
	</tr>
	<tr>
     		<td><img src="https://github.com/user-attachments/assets/dd32aeec-15c4-4740-b903-0298d031604b"></td>
       		<td><img src="https://github.com/user-attachments/assets/67187ed7-8a84-4f3b-9b40-05280c8634a2"></td>
	</tr>
</table>


## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py



# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/Adamya113/Rent_Prediction_Using_ML
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mlproj python=3.8 -y
```

```bash
conda activate mlproj
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```


## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

import dagshub
dagshub.init(repo_owner='Adamya113', repo_name='Rent_Prediction_Using_ML', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/Adamya113/Rent_Prediction_Using_ML.mlflow
```
<!-- export MLFLOW_TRACKING_USERNAME= ?? -->

<!-- export MLFLOW_TRACKING_PASSWORD= ?? -->

<!-- ``` -->



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 339712771283.dkr.ecr.eu-north-1.amazonaws.com/mlproj

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optional

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app


## About MLflow 
MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & tagging your model

<details>
	<summary>
		<h3>Acknowledgements</h3>
	</summary>
	https://mlflow.org/docs/latest/index.html
	<br>https://www.youtube.com/watch?v=pxk1Fr33-L4
	<br>https://korlakuntasaikamal10.medium.com/implementing-regression-problem-using-torch-5b5bd1a56d69
</details>



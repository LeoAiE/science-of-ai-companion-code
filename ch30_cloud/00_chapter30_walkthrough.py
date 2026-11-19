"""Chapter 30: Deploying to the Cloud: AWS, GCP, and Azure

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from Book.md; keep the book as the source.
"""

# %%
import os as _os
__file__ = _os.path.join(_os.getcwd(), "00_chapter30_walkthrough.py")

# %% [markdown]
# ## 30.1.5 When owning the machines still wins

# %%
# Linux t3.medium, US East (N. Virginia), EC2 T3 Instances page, as of November 2026.
HOURLY = 0.0418
HOURS = 730.0
rent = HOURLY * HOURS
print(f"on-demand rent for one t3.medium, {HOURS:.0f} hours: ${rent:.2f}")
print("ownership can still win if the data is not allowed to leave")
print("ownership can still win if the machine would be busy for every one of those hours")

# %% [markdown]
# ## 30.2.2 IAM users, roles, and access keys

# %% [markdown]
# `json` block from 30.2.2 IAM users, roles, and access keys (shown for reference, not executed):
#
#     {
#       "Version": "2012-10-17",
#       "Statement": [
#         {
#           "Effect": "Allow",
#           "Action": "s3:PutObject",
#           "Resource": "arn:aws:s3:::my-models-bucket/*"
#         }
#       ]
#     }

# %% [markdown]
# ## 30.2.3 The AWS CLI and where credentials land

# %% [markdown]
# `bash` block from 30.2.3 The AWS CLI and where credentials land (shown for reference, not executed):
#
#     aws configure
#     # AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
#     # AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
#     # Default region name [None]: us-east-1
#     # Default output format [None]: json

# %% [markdown]
# ## 30.2.4 boto3, the Python SDK

# %% [markdown]
# Example 30-1 proves it.

# %%
import boto3

# The installed SDK version. Pinning it in requirements would make this reproducible;
# left unpinned, whatever boto3 you have installed decides the version and region count.
print("boto3 version:", boto3.__version__)

# get_available_regions reads data bundled INSIDE boto3 (no network, no creds).
# A Session is boto3's holder for configuration and (when present) credentials.
ec2_regions = boto3.Session().get_available_regions("ec2")
print("number of EC2 regions boto3 knows:", len(ec2_regions))
print("first five regions:", ec2_regions[:5])

# Constructing a client builds a local object that KNOWS how to call S3.
# It sends no request to S3 yet, so this line works offline.
s3 = boto3.client("s3", region_name="us-east-1")
print("built S3 client, endpoint host:", s3.meta.endpoint_url)
print("S3 client region:", s3.meta.region_name)

# The us-east-1 client printed a host with no region in its name; a client
# for eu-west-1 prints a host with the region in it.
s3_eu = boto3.client("s3", region_name="eu-west-1")
print("eu-west-1 S3 endpoint:", s3_eu.meta.endpoint_url)

# %% [markdown]
# `bash` block from 30.2.4 boto3, the Python SDK (shown for reference, not executed):
#
#     aws sts get-caller-identity

# %% [markdown]
# This uploads the model artifact to a bucket; it needs your own credentials and a bucket of your own, so this book does not run it: Needs cloud credentials/SDKs: this calls a cloud service (AWS/GCP/Azure), so it is shown for reference and not executed. Run it with your own account configured.

# %%
import boto3

s3 = boto3.client("s3", region_name="us-east-1")
# Upload the local file model.joblib into bucket my-models-bucket
# under the key models/breast-cancer/model.joblib.
s3.upload_file(
    Filename="model.joblib",
    Bucket="my-models-bucket",
    Key="models/breast-cancer/model.joblib",
)
print("uploaded")

# %% [markdown]
# ## 30.3.3 Launching an instance

# %% [markdown]
# `bash` block from 30.3.3 Launching an instance (shown for reference, not executed):
#
#     # Key pair: AWS returns the private key only at creation, so save it to a file.
#     aws ec2 create-key-pair \
#       --key-name my-key \
#       --key-type rsa \
#       --key-format pem \
#       --query "KeyMaterial" \
#       --output text > my-key.pem
#
#     # Security group: admit SSH and the API from your own address only.
#     aws ec2 create-security-group \
#       --group-name breast-cancer-api \
#       --description "SSH and API access from one address"
#     aws ec2 authorize-security-group-ingress \
#       --group-id sg-0123456789abcdef0 \
#       --protocol tcp --port 22 --cidr 198.51.100.7/32
#     aws ec2 authorize-security-group-ingress \
#       --group-id sg-0123456789abcdef0 \
#       --protocol tcp --port 8000 --cidr 198.51.100.7/32
#
#     # Launch, reading the latest Amazon Linux 2023 AMI from its public parameter.
#     aws ec2 run-instances \
#       --image-id resolve:ssm:/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
#       --instance-type t3.medium \
#       --key-name my-key \
#       --security-group-ids sg-0123456789abcdef0 \
#       --count 1

# %% [markdown]
# Run this with your own AWS account (the run_instances call needs credentials; the dict it builds does not); this book does not run that call: Needs cloud credentials/SDKs: this calls a cloud service (AWS/GCP/Azure), so it is shown for reference and not executed. Run it with your own account configured.

# %%
import boto3

ec2 = boto3.client("ec2", region_name="us-east-1")

params = {
    "ImageId": "ami-0abcdef1234567890",
    "InstanceType": "t3.medium",
    "KeyName": "my-key",
    "SecurityGroupIds": ["sg-0123456789abcdef0"],
    "MinCount": 1,
    "MaxCount": 1,
}
print("request I will send:", params)

response = ec2.run_instances(**params)
instance_id = response["Instances"][0]["InstanceId"]
print("launched instance:", instance_id)

# %% [markdown]
# Example 30-3. Find the AMI, create the key pair and the security group, then launch and terminate Example 30-2's instance, all against moto.

# %%
import os
from unittest import mock

import boto3
from moto import mock_aws

# Dummy credentials, as moto's documentation advises, so that boto3 never tries
# real ones. patch.dict sets them for the with block only, then puts back
# whatever the four variables held before.
FAKE = {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing", "AWS_DEFAULT_REGION": "us-east-1"}
AL2023 = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
MY_ADDRESS = "198.51.100.7/32"   # stands for your own computer's public IPv4 address
env_before = {name: os.environ.get(name) for name in FAKE}

with mock.patch.dict(os.environ, FAKE), mock_aws():
    ssm_client = boto3.client("ssm", region_name="us-east-1")
    ec2_client = boto3.client("ec2", region_name="us-east-1")

    # 1. The AMI. On AWS this public parameter holds the latest Amazon Linux 2023
    #    image for x86_64; moto answers from a copy of it that ships with moto.
    ami = ssm_client.get_parameter(Name=AL2023)["Parameter"]["Value"]
    print("AMI ID: starts with 'ami-':", ami.startswith("ami-"), "| length:", len(ami))

    # 2. The key pair. The private key comes back in this one response; a real
    #    script writes key["KeyMaterial"] to my-key.pem at this point.
    key = ec2_client.create_key_pair(KeyName="my-key")
    print("key pair:", key["KeyName"], "| private key returned:",
          key["KeyMaterial"].startswith("-----BEGIN"))

    # 3. The security group: SSH on port 22 and the API on port 8000, from one address.
    group_id = ec2_client.create_security_group(
        GroupName="breast-cancer-api", Description="SSH and the API port")["GroupId"]
    ec2_client.authorize_security_group_ingress(GroupId=group_id, IpPermissions=[
        {"IpProtocol": "tcp", "FromPort": port, "ToPort": port,
         "IpRanges": [{"CidrIp": MY_ADDRESS}]} for port in (22, 8000)])
    rules = ec2_client.describe_security_groups(
        GroupIds=[group_id])["SecurityGroups"][0]["IpPermissions"]
    print("security group:", group_id[:3] + "...",
          "| inbound ports:", sorted(rule["FromPort"] for rule in rules),
          "| from:", sorted({rule["IpRanges"][0]["CidrIp"] for rule in rules}))

    # 4. Example 30-2's request, now naming the AMI, key pair, and group of steps 1 to 3.
    request = {"ImageId": ami, "InstanceType": "t3.medium", "KeyName": "my-key",
               "SecurityGroupIds": [group_id], "MinCount": 1, "MaxCount": 1}
    instance = ec2_client.run_instances(**request)["Instances"][0]
    print("launched:", instance["InstanceId"][:2] + "...",
          "| type:", instance["InstanceType"], "| key:", instance["KeyName"],
          "| state:", instance["State"]["Name"])

    # 5. Terminate it. On a real account, the instance's charges stop here.
    ended = ec2_client.terminate_instances(InstanceIds=[instance["InstanceId"]])
    print("after terminate:", ended["TerminatingInstances"][0]["CurrentState"]["Name"])

print("the four variables are as before the block:",
      {name: os.environ.get(name) for name in FAKE} == env_before)

# %% [markdown]
# ## 30.3.4 Connecting and setting up

# %% [markdown]
# `bash` block from 30.3.4 Connecting and setting up (shown for reference, not executed):
#
#     chmod 400 my-key.pem
#     ssh -i my-key.pem ec2-user@203.0.113.42

# %% [markdown]
# `bash` block from 30.3.4 Connecting and setting up (shown for reference, not executed):
#
#     sudo yum update -y
#     sudo yum install docker
#     sudo service docker start
#     sudo usermod -a -G docker ec2-user
#     # log out and back in, so the new docker group applies, then check:
#     docker info

# %% [markdown]
# ## 30.4.2 Package the model once

# %% [markdown]
# This runs fully offline.

# %%
import os
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Real dataset: 569 tumors, 30 numeric features, label 0 = malignant, 1 = benign.
data = load_breast_cancer()
X, y = data.data, data.target
print("dataset shape:", X.shape, "labels:", set(y.tolist()))

# A pipeline bundles preprocessing + model so the deployed artifact is self-contained:
# the scaler's fitted means/scales travel WITH the classifier.
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=10000),
)
model.fit(X, y)
print("train accuracy:", round(model.score(X, y), 4))

# Freeze to disk with joblib; scikit-learn's persistence guide describes it as
# more efficient than pickle for large models and large NumPy arrays.
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
out = os.path.join(HERE, "model.joblib")
joblib.dump(model, out)
print("saved artifact:", os.path.basename(out), "size(bytes):", os.path.getsize(out))

# Sanity prediction on the first tumor. predict_proba gives [P(malignant), P(benign)].
first = X[:1]
proba = model.predict_proba(first)[0]
print("first sample P(malignant):", round(float(proba[0]), 4),
      "P(benign):", round(float(proba[1]), 4))

# %% [markdown]
# Example 30-5. Upload Example 30-4's artifact to S3 against moto, overwrite it, and keep both models with versioning.

# %%
# Reads out, the path where Example 30-4 saved model.joblib, and its data X and y.
import os
import tempfile
from unittest import mock

import boto3
import joblib
from moto import mock_aws
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FAKE = {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing", "AWS_DEFAULT_REGION": "us-east-1"}
KEY = "models/breast-cancer/model.joblib"    # the key of the upload in 30.2.4
with open(out, "rb") as fh:
    bytes_v1 = fh.read()                     # the first model, byte for byte

with mock.patch.dict(os.environ, FAKE), mock_aws(), tempfile.TemporaryDirectory() as tmp:
    s3_client = boto3.client("s3", region_name="us-east-1")
    s3_client.create_bucket(Bucket="my-models-bucket")

    # The upload of 30.2.4, answered by moto. Then read the size and the bytes back.
    s3_client.upload_file(Filename=out, Bucket="my-models-bucket", Key=KEY)
    size = s3_client.head_object(Bucket="my-models-bucket", Key=KEY)["ContentLength"]
    print("stored bytes:", size, "| same as the local artifact:", size == len(bytes_v1))
    body = s3_client.get_object(Bucket="my-models-bucket", Key=KEY)["Body"].read()
    print("downloaded bytes equal the artifact:", body == bytes_v1)

    # A second model, the same pipeline with C=0.01, saved in the temporary folder
    # and uploaded to the same key.
    model_v2 = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=10000))
    path_v2 = os.path.join(tmp, "model_v2.joblib")
    joblib.dump(model_v2.fit(X, y), path_v2)
    s3_client.upload_file(Filename=path_v2, Bucket="my-models-bucket", Key=KEY)
    stored = s3_client.get_object(Bucket="my-models-bucket", Key=KEY)["Body"].read()
    print("after a second upload to the same key, the first model is still stored:",
          stored == bytes_v1)

    # The same two uploads into a bucket that keeps every version.
    s3_client.create_bucket(Bucket="my-models-bucket-versioned")
    s3_client.put_bucket_versioning(Bucket="my-models-bucket-versioned",
                                    VersioningConfiguration={"Status": "Enabled"})
    for artifact in (out, path_v2):
        s3_client.upload_file(Filename=artifact, Bucket="my-models-bucket-versioned", Key=KEY)
    versions = s3_client.list_object_versions(
        Bucket="my-models-bucket-versioned", Prefix=KEY)["Versions"]
    older_id = [v["VersionId"] for v in versions if not v["IsLatest"]][0]
    older = s3_client.get_object(Bucket="my-models-bucket-versioned", Key=KEY,
                                 VersionId=older_id)["Body"].read()
    print("with versioning on:", len(versions), "versions | the older one is the first model:",
          older == bytes_v1)

# %% [markdown]
# ## 30.4.3 Wrap it in a web app

# %% [markdown]
# It runs fully offline.

# %%
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_breast_cancer

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
model = joblib.load(os.path.join(HERE, "model.joblib"))

app = FastAPI(title="breast-cancer-classifier")

class Tumor(BaseModel):
    # 30 numeric features, the exact order load_breast_cancer produces.
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(t: Tumor):
    proba = model.predict_proba([t.features])[0]
    return {
        "malignant_probability": round(float(proba[0]), 4),
        "label": "malignant" if proba[0] >= 0.5 else "benign",
    }

if __name__ == "__main__":
    from fastapi.testclient import TestClient
    client = TestClient(app)

    print("GET /health ->", client.get("/health").json())

    data = load_breast_cancer()
    sample = data.data[0].tolist()   # first tumor, known malignant
    r = client.post("/predict", json={"features": sample})
    print("POST /predict ->", r.status_code, r.json())

# %% [markdown]
# ## 30.4.4 Path A: a container on EC2

# %% [markdown]
# `dockerfile` block from 30.4.4 Path A: a container on EC2 (shown for reference, not executed):
#
#     FROM python:3.13.18-slim
#     WORKDIR /app
#     COPY requirements.txt .
#     RUN pip install --no-cache-dir -r requirements.txt
#     COPY model.joblib serve_local.py ./
#     EXPOSE 8000
#     CMD ["uvicorn", "serve_local:app", "--host", "0.0.0.0", "--port", "8000"]

# %% [markdown]
# `bash` block from 30.4.4 Path A: a container on EC2 (shown for reference, not executed):
#
#     # Build the image locally (Example 28-9 explains each Dockerfile instruction).
#     docker build -t breast-cancer-api .
#
#     # Create the private repository once, then tag and push to ECR
#     # (ACCOUNT is your 12-digit account number).
#     aws ecr create-repository --repository-name breast-cancer-api --region us-east-1
#     aws ecr get-login-password --region us-east-1 | \
#       docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
#     docker tag breast-cancer-api ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/breast-cancer-api:latest
#     docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/breast-cancer-api:latest
#
#     # On the EC2 instance (via SSH), with Docker set up as in 30.3.4, an instance role
#     # that includes AmazonEC2ContainerRegistryPullOnly, and the AWS CLI installed:
#     aws ecr get-login-password --region us-east-1 | \
#       docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
#     docker run -d -p 8000:8000 ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/breast-cancer-api:latest

# %% [markdown]
# ## 30.4.5 Path B: a serverless function on Lambda

# %% [markdown]
# Illustrative Lambda handler and deployment:

# %%
# lambda_function.py
import json
import joblib

# Loaded once per warm container, reused across invocations.
model = joblib.load("model.joblib")

def lambda_handler(event, context):
    body = json.loads(event["body"])
    proba = model.predict_proba([body["features"]])[0]
    return {
        "statusCode": 200,
        "body": json.dumps({
            "malignant_probability": round(float(proba[0]), 4),
            "label": "malignant" if proba[0] >= 0.5 else "benign",
        }),
    }

# %% [markdown]
# `bash` block from 30.4.5 Path B: a serverless function on Lambda (shown for reference, not executed):
#
#     # Package the handler + model + deps into a zip, then create the function.
#     aws lambda create-function \
#       --function-name breast-cancer-api \
#       --runtime python3.12 \
#       --handler lambda_function.lambda_handler \
#       --zip-file fileb://deploy.zip \
#       --role arn:aws:iam::ACCOUNT:role/lambda-exec-role \
#       --timeout 10 --memory-size 512

# %% [markdown]
# `bash` block from 30.4.5 Path B: a serverless function on Lambda (shown for reference, not executed):
#
#     # The container-image route: same function, room for real dependency stacks.
#     aws lambda create-function \
#       --function-name breast-cancer-api \
#       --package-type Image \
#       --code ImageUri=ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/breast-cancer-lambda:1.0 \
#       --role arn:aws:iam::ACCOUNT:role/lambda-exec-role \
#       --timeout 10 --memory-size 512

# %% [markdown]
# ## 30.4.6 Path C: a managed endpoint on Amazon SageMaker AI

# %% [markdown]
# `bash` block from 30.4.6 Path C: a managed endpoint on Amazon SageMaker AI (shown for reference, not executed):
#
#     pip install "sagemaker==2.*"   # the 2.x interface this sketch uses
#     tar -czf model.tar.gz model.joblib
#     aws s3 cp model.tar.gz s3://my-models-bucket/models/breast-cancer/model.tar.gz

# %% [markdown]
# A minimal inference.py needs only model_fn:

# %%
# inference.py
import os
import joblib

def model_fn(model_dir):
    # model_dir holds the files SageMaker AI took from model.tar.gz.
    return joblib.load(os.path.join(model_dir, "model.joblib"))

# %% [markdown]
# Illustrative SageMaker AI deploy; this book does not run it, because it needs your AWS account and the SageMaker Python SDK 2.x, which this book does not install: Needs cloud credentials/SDKs: this calls a cloud service (AWS/GCP/Azure), so it is shown for reference and not executed. Run it with your own account configured.

# %%
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sklearn.datasets import load_breast_cancer

model = SKLearnModel(
    model_data="s3://my-models-bucket/models/breast-cancer/model.tar.gz",
    role="arn:aws:iam::ACCOUNT:role/sagemaker-exec-role",
    entry_point="inference.py",   # defines model_fn; the default predict_fn calls model.predict
    framework_version="1.2-1",    # an example tag, not a claim that this string is current; confirm the scikit-learn image in the SageMaker docs before you deploy
)

predictor = model.deploy(
    instance_type="ml.m5.large",
    initial_instance_count=1,
    endpoint_name="breast-cancer-endpoint",
)

# The default SKLearn predictor serializes NumPy, so send a 2D array, not a dict.
# (A custom inference.py's input_fn could accept JSON instead.) The model expects
# all 30 features, so build the request from the real dataset rather than typing them.
first_tumor = load_breast_cancer().data[:1]   # shape (1, 30): the first tumor's 30 features
result = predictor.predict(first_tumor)
print(result)

# %% [markdown]
# ## 30.5.1 Account, project, and the identity boundary

# %% [markdown]
# `bash` block from 30.5.1 Account, project, and the identity boundary (shown for reference, not executed):
#
#     gcloud auth login
#     gcloud config set project my-ml-project-2026
#     gcloud config set run/region us-central1

# %% [markdown]
# ## 30.5.2 Compute Engine: the VM path

# %% [markdown]
# `bash` block from 30.5.2 Compute Engine: the VM path (shown for reference, not executed):
#
#     gcloud compute instances create bc-vm \
#         --machine-type=e2-medium \
#         --image-family=debian-12 --image-project=debian-cloud \
#         --zone=us-central1-a

# %% [markdown]
# ## 30.5.3 Artifact Registry: where the image lives

# %% [markdown]
# `bash` block from 30.5.3 Artifact Registry: where the image lives (shown for reference, not executed):
#
#     gcloud artifacts repositories create ml-images \
#         --repository-format=docker --location=us-central1
#
#     gcloud auth configure-docker us-central1-docker.pkg.dev
#
#     docker tag breast-cancer-api \
#         us-central1-docker.pkg.dev/my-ml-project-2026/ml-images/breast-cancer-api:v1
#     docker push \
#         us-central1-docker.pkg.dev/my-ml-project-2026/ml-images/breast-cancer-api:v1

# %% [markdown]
# ## 30.5.4 Cloud Run: one command from image to URL

# %% [markdown]
# `bash` block from 30.5.4 Cloud Run: one command from image to URL (shown for reference, not executed):
#
#     gcloud run deploy bc-service \
#         --image us-central1-docker.pkg.dev/my-ml-project-2026/ml-images/breast-cancer-api:v1 \
#         --region us-central1 \
#         --port 8000 \
#         --allow-unauthenticated

# %% [markdown]
# ## 30.5.5 Vertex AI: the managed ML endpoint

# %% [markdown]
# Run this sketch with google-cloud-aiplatform installed and your project configured: Needs cloud credentials/SDKs: this calls a cloud service (AWS/GCP/Azure), so it is shown for reference and not executed. Run it with your own account configured.

# %%
from google.cloud import aiplatform

aiplatform.init(project="my-ml-project-2026", location="us-central1")

model = aiplatform.Model.upload(
    display_name="bc-logreg",
    serving_container_image_uri=(
        "us-central1-docker.pkg.dev/my-ml-project-2026/ml-images/breast-cancer-api:v1"
    ),
    serving_container_ports=[8000],
    serving_container_predict_route="/predict",   # where Vertex forwards predictions
    serving_container_health_route="/health",     # where Vertex polls for liveness
)

endpoint = model.deploy(
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=3,
)
print("Endpoint:", endpoint.resource_name)

# %% [markdown]
# ## 30.6.1 Subscription, resource group, and identity

# %% [markdown]
# `bash` block from 30.6.1 Subscription, resource group, and identity (shown for reference, not executed):
#
#     az login
#     az group create --name ml-rg --location eastus

# %% [markdown]
# ## 30.6.2 Virtual Machines, Container Apps, and Azure ML

# %% [markdown]
# `bash` block from 30.6.2 Virtual Machines, Container Apps, and Azure ML (shown for reference, not executed):
#
#     az vm create \
#         --resource-group ml-rg \
#         --name bc-vm \
#         --image Ubuntu2204 \
#         --size Standard_B2ls_v2 \
#         --admin-username azureuser \
#         --generate-ssh-keys

# %% [markdown]
# `bash` block from 30.6.2 Virtual Machines, Container Apps, and Azure ML (shown for reference, not executed):
#
#     az acr create --resource-group ml-rg --name mlregistry2026 --sku Basic
#     az acr login --name mlregistry2026
#
#     docker tag breast-cancer-api mlregistry2026.azurecr.io/breast-cancer-api:v1
#     docker push mlregistry2026.azurecr.io/breast-cancer-api:v1

# %% [markdown]
# `bash` block from 30.6.2 Virtual Machines, Container Apps, and Azure ML (shown for reference, not executed):
#
#     az containerapp env create \
#         --name ml-env --resource-group ml-rg --location eastus
#
#     az identity create --name ml-identity --resource-group ml-rg
#     az identity show --name ml-identity --resource-group ml-rg \
#         --query id --output tsv
#
#     az containerapp create \
#         --name bc-service \
#         --resource-group ml-rg \
#         --environment ml-env \
#         --image mlregistry2026.azurecr.io/breast-cancer-api:v1 \
#         --target-port 8000 \
#         --ingress external \
#         --user-assigned <IDENTITY_ID> \
#         --registry-identity <IDENTITY_ID> \
#         --registry-server mlregistry2026.azurecr.io

# %% [markdown]
# Run this sketch with azure-ai-ml and azure-identity installed: Needs cloud credentials/SDKs: this calls a cloud service (AWS/GCP/Azure), so it is shown for reference and not executed. Run it with your own account configured.

# %%
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint, ManagedOnlineDeployment, Model, Environment,
)
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<sub-id>",
    resource_group_name="ml-rg",
    workspace_name="ml-workspace",
)

endpoint = ManagedOnlineEndpoint(name="bc-endpoint", auth_mode="key")
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="bc-endpoint",
    model=Model(path="./model"),
    environment=Environment(
        image="mlregistry2026.azurecr.io/breast-cancer-api:v1",
        # Your own container must say where Azure ML probes it and sends requests.
        inference_config={
            "liveness_route": {"port": 8000, "path": "/health"},
            "readiness_route": {"port": 8000, "path": "/health"},
            "scoring_route": {"port": 8000, "path": "/predict"},
        },
    ),
    instance_type="Standard_D2as_v4",
    instance_count=1,
)
ml_client.online_deployments.begin_create_or_update(deployment).result()

endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# %% [markdown]
# ## 30.6.3 Managed Kubernetes: EKS, GKE, and AKS

# %% [markdown]
# `yaml` block from 30.6.3 Managed Kubernetes: EKS, GKE, and AKS (shown for reference, not executed):
#
#     apiVersion: serving.kserve.io/v1beta1
#     kind: InferenceService
#     metadata:
#       name: bc-service
#     spec:
#       predictor:
#         containers:
#           - name: kserve-container
#             image: mlregistry2026.azurecr.io/breast-cancer-api:v1
#             ports:
#               - containerPort: 8000
#                 protocol: TCP

# %%
import yaml

# The InferenceService manifest of 30.6.3, held as text exactly as you would save it.
bc_service_manifest = """
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: bc-service
spec:
  predictor:
    containers:
      - name: kserve-container
        image: mlregistry2026.azurecr.io/breast-cancer-api:v1
        ports:
          - containerPort: 8000
            protocol: TCP
"""

# safe_load turns the indentation into nested dicts and lists. It resolves only
# basic YAML tags, which its docstring calls safe for untrusted input.
isvc = yaml.safe_load(bc_service_manifest)
isvc_container = isvc["spec"]["predictor"]["containers"][0]

# The fields a typo would break: the kind of resource, its name, the image, the port.
print(isvc["kind"], isvc["metadata"]["name"])
print(isvc_container["name"], isvc_container["image"])
print([p["containerPort"] for p in isvc_container["ports"]])

# %% [markdown]
# `bash` block from 30.6.3 Managed Kubernetes: EKS, GKE, and AKS (shown for reference, not executed):
#
#     kubectl apply -f bc-service.yaml

# %% [markdown]
# ## 30.6.4 Renting a foundation model: Bedrock, Vertex AI, and Microsoft Foundry

# %% [markdown]
# Run this with your own AWS account; the call needs your credentials, so this book does not run it: Needs cloud credentials/SDKs: this calls a cloud service (AWS/GCP/Azure), so it is shown for reference and not executed. Run it with your own account configured.

# %%
import boto3

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
response = bedrock.converse(
    modelId="MODEL_ID",   # pick a model ID from the Bedrock console's catalog
    messages=[{"role": "user",
               "content": [{"text": "In one sentence, what does a load balancer do?"}]}],
)
print(response["output"]["message"]["content"][0]["text"])
print(response["usage"])   # inputTokens / outputTokens: the units you are billed in

# %% [markdown]
# ## 30.7.2 Storage classes and the price/latency tradeoff

# %% [markdown]
# storing data is cheap, but shipping it out is a charge that can surprise you unless you price it first.

# %%
# US East (N. Virginia) list prices in USD per GB-month, AWS Price List, November 2026.
PRICE_PER_GB_MONTH = {
    "Standard": 0.023,                 # hot: frequent access, millisecond reads
    "Standard-IA": 0.0125,             # infrequent access: cheaper store, retrieval fee
    "Glacier Deep Archive": 0.00099,   # archive: cheapest store, hours to restore
}
# Per-GB retrieval fee when you READ data back from a colder class.
PRICE_PER_GB_RETRIEVAL = {"Standard": 0.0, "Standard-IA": 0.01, "Glacier Deep Archive": 0.02}
# Internet egress beyond the global free tier, first 10 TB a month.
PRICE_PER_GB_EGRESS = 0.09

def report(footprint_gb, gb_read_back, gb_egress):
    print(f"Footprint: {footprint_gb} GB stored; "
          f"{gb_read_back} GB read back; {gb_egress} GB egress to internet\n")
    header = f"{'Class':<22}{'Store/mo':>12}{'Retrieval':>12}{'Egress':>10}{'Total/mo':>12}"
    print(header); print("-" * len(header))
    for cls in PRICE_PER_GB_MONTH:
        store = footprint_gb * PRICE_PER_GB_MONTH[cls]
        retr = gb_read_back * PRICE_PER_GB_RETRIEVAL[cls]
        egr = gb_egress * PRICE_PER_GB_EGRESS
        print(f"{cls:<22}{store:>11.2f}{retr:>12.2f}{egr:>10.2f}{store+retr+egr:>12.2f}")

# 500 GB of datasets + artifacts; pull 50 GB back for a retrain; ship 20 GB to a client.
report(footprint_gb=500, gb_read_back=50, gb_egress=20)

# %% [markdown]
# ## 30.7.4 A versioned key so the previous artifact survives

# %%
def versioned_key(name, version):
    return f"models/{name}/{version}/model.joblib"

def put_once(store, key, payload):
    if key in store:
        raise ValueError(f"refusing to overwrite {key}")
    store[key] = payload
    return key

POINTER = "models/breast-cancer/current"   # P: the key of the live version

def promote(store, key):
    if key not in store:
        raise KeyError(f"nothing stored at {key}")
    previous = store.get(POINTER)   # remember what was live, for a rollback
    store[POINTER] = key            # rewrite only the pointer, no artifact
    return previous

def rollback(store, previous):
    store[POINTER] = previous       # point back; no artifact moves

store = {}
first = put_once(store, versioned_key("breast-cancer", "v1"), b"artifact-v1")
second = put_once(store, versioned_key("breast-cancer", "v2"), b"artifact-v2")
print(first)
print(second)
print(sorted(store))
try:
    put_once(store, first, b"bad")
except ValueError as err:
    print(err)
promote(store, first)
print("current ->", store[POINTER])
previous = promote(store, second)   # v2 goes live, v1 is remembered
print("current ->", store[POINTER])
rollback(store, previous)           # v2 misbehaves: point back at v1
print("current ->", store[POINTER], store[store[POINTER]])

# %% [markdown]
# ## 30.8.3 A cost model you can run

# %% [markdown]
# The instance price is the Linux t3.medium on-demand rate in US East (N. Virginia) from the EC2 T3 Instances page (Amazon Web Services, 2026a), as of November 2026. The storage and egress rates below are the US East list prices for S3 Standard storage and for data transfer out to the internet in the AWS Price List (Amazon Web Services, 2026t), as of November 2026.

# %%
# ---- On-demand price for one small serving instance -------------------------
# Linux t3.medium, US East (N. Virginia), EC2 T3 Instances page, as of
# November 2026: $0.0418/hour, 2 vCPUs, 4 GiB.
ON_DEMAND_HOURLY = 0.0418
HOURS_PER_MONTH = 730.0  # the cloud-billing convention (365*24/12)

# Same page: 1-year partial-upfront effective hourly for this instance.
RESERVED_HOURLY = 0.025
# Spot discount is an assumption so the arithmetic is visible, not a quoted rate.
SPOT_DISCOUNT = 0.70
spot_hourly = ON_DEMAND_HOURLY * (1 - SPOT_DISCOUNT)

# ---- Storage and egress: US East list prices, AWS Price List, November 2026 -
STORAGE_GB, STORAGE_PRICE = 50.0, 0.023
EGRESS_GB, EGRESS_PRICE = 100.0, 0.09
storage_cost = STORAGE_GB * STORAGE_PRICE
egress_cost = EGRESS_GB * EGRESS_PRICE
fixed_side = storage_cost + egress_cost

def monthly_compute(hourly):
    return hourly * HOURS_PER_MONTH

print("=== Always-on serving: one instance, 24/7 for a month ===")
print(f"storage: {STORAGE_GB:.0f} GB x ${STORAGE_PRICE}/GB-mo = ${storage_cost:.2f}")
print(f"egress:  {EGRESS_GB:.0f} GB x ${EGRESS_PRICE}/GB    = ${egress_cost:.2f}\n")
for name, hourly in [("on-demand", ON_DEMAND_HOURLY),
                     ("1-yr reserved", RESERVED_HOURLY),
                     ("spot", spot_hourly)]:
    compute = monthly_compute(hourly)
    print(f"{name:>14}: ${hourly:.5f}/hr x 730 = ${compute:7.2f} compute "
          f"+ ${fixed_side:6.2f} fixed = ${compute + fixed_side:7.2f}/mo")

# ---- Break-even: always-on managed endpoint vs per-request serverless -------
# Fixed monthly cost (always-on) vs per-unit cost (serverless), the Chapter 28
# build-versus-buy crossover in cloud form. Egress is dropped from both sides
# here because both options pay the same egress, so it cancels at the crossover.
always_on_fixed = monthly_compute(ON_DEMAND_HOURLY) + storage_cost
# $0.50 per million is a REQUEST-ONLY stand-in that keeps the mechanics visible;
# a real Lambda bill adds the GB-second compute charge (30.4.5: most of the total).
SERVERLESS_PER_REQUEST = 0.50 / 1_000_000
breakeven = always_on_fixed / SERVERLESS_PER_REQUEST
print("\n=== Break-even: always-on endpoint vs per-request serverless ===")
print(f"always-on fixed monthly cost: ${always_on_fixed:.2f}")
print(f"serverless price per request: ${SERVERLESS_PER_REQUEST:.7f}")
print(f"break-even volume: {breakeven:,.0f} requests/month "
      f"(~{breakeven/HOURS_PER_MONTH/3600:.1f} req/s)")
for n in (1_000_000, 200_000_000):
    serverless = n * SERVERLESS_PER_REQUEST
    winner = "serverless" if serverless < always_on_fixed else "always-on"
    print(f"  at {n:>13,.0f} req/mo: serverless ${serverless:8.2f} vs "
          f"always-on ${always_on_fixed:7.2f} -> {winner} wins")

# Redo the break-even with 30.4.5's Lambda price before the gateway (~$1.87 per
# million: request charge + GB-second compute) and watch the crossover move.
LAMBDA_PER_REQUEST = 1.87 / 1_000_000
print(f"with 30.4.5's Lambda price before the gateway (~$1.87/M): break-even "
      f"{always_on_fixed / LAMBDA_PER_REQUEST:,.0f} requests/month")
# Add the HTTP API's $1.00 per million in front of Lambda: 30.4.5's all-in
# price of ~$2.87 per million moves the crossover again.
HTTP_API_ALL_IN_PER_REQUEST = 2.87 / 1_000_000
print(f"with 30.4.5's all-in price behind an HTTP API (~$2.87/M): break-even "
      f"{always_on_fixed / HTTP_API_ALL_IN_PER_REQUEST:,.0f} requests/month")

# %% [markdown]
# ## 30.9.1 CI/CD to the cloud

# %% [markdown]
# `yaml` block from 30.9.1 CI/CD to the cloud (shown for reference, not executed):
#
#     name: build-and-deploy
#     on:
#       push:
#         branches: [main]
#     jobs:
#       deploy:
#         runs-on: ubuntu-latest
#         permissions:
#           contents: 'read'
#           id-token: 'write'
#         env:
#           SERVICE: breast-cancer-api
#           REGION: us-central1
#           IMAGE: us-central1-docker.pkg.dev/my-ml-project-2026/ml-images/breast-cancer-api
#         steps:
#           - uses: actions/checkout@v7
#           - uses: google-github-actions/auth@v3
#             with:
#               workload_identity_provider: 'projects/123456789/locations/global/workloadIdentityPools/my-pool/providers/my-provider'
#               service_account: 'my-service-account@my-ml-project-2026.iam.gserviceaccount.com'
#           - uses: google-github-actions/setup-gcloud@v3
#           - name: Build and push
#             run: |
#               gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
#               docker build -t "$IMAGE:${{ github.sha }}" .   # tag with the commit SHA
#               docker push "$IMAGE:${{ github.sha }}"
#           - name: Deploy new revision (parked at 0% traffic)
#             run: |
#               gcloud run deploy "$SERVICE" \
#                 --image "$IMAGE:${{ github.sha }}" \
#                 --region "$REGION" \
#                 --no-traffic --tag "sha-${{ github.sha }}"

# %% [markdown]
# ## 30.9.4 A canary rollout you can run

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)

def fit(C):  # C is the inverse regularization strength
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000)).fit(X_tr, y_tr)

incumbent = fit(C=0.01)   # heavily regularized, currently in production
candidate = fit(C=1.0)    # the "new" model we want to roll out
acc_incumbent = incumbent.score(X_te, y_te)
acc_candidate = candidate.score(X_te, y_te)
print(f"incumbent  (C=0.01) holdout accuracy: {acc_incumbent:.4f}")
print(f"candidate  (C=1.00) holdout accuracy: {acc_candidate:.4f}")

STAGES = [0.05, 0.25, 0.50, 1.00]   # fraction of traffic to the candidate
MARGIN = 0.0                        # required lead; 0.0 means a tie passes
# The holdout rows play live traffic. Shuffle them once; at each stage the
# first rows of this order go to the candidate and the rest to the incumbent.
order = np.random.default_rng(42).permutation(len(y_te))

def rollout(new, old):
    for frac in STAGES:
        k = round(frac * len(order))
        to_new, to_old = order[:k], order[k:]
        acc_new = new.score(X_te[to_new], y_te[to_new])
        # Score the old model on the rows it still serves; at 100 percent it
        # serves none, so its full holdout accuracy is the baseline.
        acc_old = (old.score(X_te[to_old], y_te[to_old]) if len(to_old)
                   else old.score(X_te, y_te))
        ok = acc_new >= acc_old + MARGIN
        print(f"  stage {frac*100:5.1f}% ({k:3d} rows): candidate {acc_new:.4f}"
              f" vs incumbent {acc_old:.4f} -> {'advance' if ok else 'ROLLBACK'}")
        if not ok:
            return "regression detected -> roll back to incumbent."
    return "PROMOTE to 100%."

print("\nStaged canary rollout:")
print("Decision:", rollout(candidate, incumbent))
print("\nThe same rollout with the C=0.01 model as the candidate:")
print("Decision:", rollout(incumbent, candidate))

# %% [markdown]
# ## 30.9.5 A deploy specification you can check

# %%
def check_spec(spec):
    required = ("image", "port", "health_path", "execution_role")
    missing = [field for field in required if not spec.get(field)]
    if missing:
        raise ValueError("missing " + ", ".join(missing))
    return spec

spec = {
    "image": "breast-cancer-api:1.0",
    "port": 8080,
    "health_path": "/health",
    "execution_role": "ecsTaskExecutionRole",
}
print(check_spec(spec))
blank = dict(spec)
blank["health_path"] = ""
try:
    check_spec(blank)
except ValueError as err:
    print(err)

# %% [markdown]
# ## 30.10.1 The local pipeline, end to end

# %% [markdown]
# code/ch30_cloud/lab_build_serve.py runs the training and the test end to end, and code/ch30_cloud/app.py is the serving file, checked in exactly as the container copies it.

# %%
import os, joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=5000)).fit(X_tr, y_tr)
joblib.dump(model, os.path.join(HERE, "model.joblib"))
print(f"trained pipeline, holdout accuracy = {model.score(X_te, y_te):.4f}")
print(f"expects {X.shape[1]} features; saved to model.joblib")

# %% [markdown]
# The artifact is done; now the file that serves it.

# %%
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
served = joblib.load(os.path.join(HERE, "model.joblib"))

app = FastAPI(title="breast-cancer-classifier")

class Features(BaseModel):
    values: list[float]   # 30 numbers, one per feature

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(f: Features):
    # class 1 is "benign" in load_breast_cancer (target_names[1]).
    proba = round(float(served.predict_proba([f.values])[0][1]), 4)
    return {"probability_benign": proba, "label": int(proba >= 0.5)}

# %% [markdown]
# The test harness, Example 30-16, exercises it with FastAPI's TestClient, a testing tool that calls the app's routes directly in the same Python process, no server and no network, which is exactly how you verify serving logic offline.
#
# Example 30-16. Exercise Example 30-15's app in process with TestClient: a health check, a prediction on a holdout tumor from Example 30-14's split, and the warm latency after one throwaway request.

# %%
import time
from app import app
from fastapi.testclient import TestClient

client = TestClient(app)
sample = X_te[0].tolist()          # a real holdout tumor from Example 30-14's split

# Warm-up: the FIRST TestClient request pays one-time startup costs (it imports
# the async plumbing behind TestClient) that say nothing about the model. Pay
# them outside the timed block.
client.post("/predict", json={"values": sample})

t0 = time.perf_counter()
resp = client.post("/predict", json={"values": sample})
latency_ms = (time.perf_counter() - t0) * 1000
print(f"health check: {client.get('/health').json()}")
print(f"POST /predict -> {resp.json()}")
print(f"true label for this sample: {int(y_te[0])}")
print(f"warm in-process latency: {latency_ms:.1f} ms")

# %% [markdown]
# ## 30.10.2 The shared Dockerfile and the one-command deploys

# %% [markdown]
# `dockerfile` block from 30.10.2 The shared Dockerfile and the one-command deploys (shown for reference, not executed):
#
#     FROM python:3.13.18-slim
#     WORKDIR /app
#     COPY requirements.txt .
#     RUN pip install --no-cache-dir -r requirements.txt
#     COPY model.joblib app.py ./
#     ENV PORT=8080
#     CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]

# %% [markdown]
# `bash` block from 30.10.2 The shared Dockerfile and the one-command deploys (shown for reference, not executed):
#
#     aws iam create-role --role-name ecsTaskExecutionRole \
#       --assume-role-policy-document '{"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
#           "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'
#     aws iam attach-role-policy --role-name ecsTaskExecutionRole \
#       --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
#
#     aws iam create-role --role-name ecsInfrastructureRoleForExpressServices \
#       --assume-role-policy-document '{"Version": "2012-10-17", "Statement": [{
#           "Sid": "AllowAccessInfrastructureForECSExpressServices", "Effect": "Allow",
#           "Principal": {"Service": "ecs.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'
#     aws iam attach-role-policy --role-name ecsInfrastructureRoleForExpressServices \
#       --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices

# %% [markdown]
# `bash` block from 30.10.2 The shared Dockerfile and the one-command deploys (shown for reference, not executed):
#
#     aws ecs create-express-gateway-service \
#       --service-name breast-cancer-api \
#       --primary-container '{"image":"<acct>.dkr.ecr.us-east-1.amazonaws.com/breast-cancer-api:1.0",
#           "containerPort":8080}' \
#       --execution-role-arn arn:aws:iam::<acct>:role/ecsTaskExecutionRole \
#       --infrastructure-role-arn arn:aws:iam::<acct>:role/ecsInfrastructureRoleForExpressServices \
#       --health-check-path /health

# %% [markdown]
# `bash` block from 30.10.2 The shared Dockerfile and the one-command deploys (shown for reference, not executed):
#
#     gcloud run deploy breast-cancer-api \
#       --image us-central1-docker.pkg.dev/my-ml-project-2026/ml-images/breast-cancer-api:1.0 \
#       --region us-central1 --allow-unauthenticated

# %% [markdown]
# `bash` block from 30.10.2 The shared Dockerfile and the one-command deploys (shown for reference, not executed):
#
#     az containerapp create \
#       --name breast-cancer-api --resource-group ml-rg \
#       --environment ml-env \
#       --image mlregistry2026.azurecr.io/breast-cancer-api:1.0 \
#       --target-port 8080 --ingress external \
#       --user-assigned <IDENTITY_ID> --registry-identity <IDENTITY_ID> \
#       --registry-server mlregistry2026.azurecr.io

# %% [markdown]
# ## 30.11 Interview questions and coding exercises

# %% [markdown]
# Starter code for coding exercise 1.

# %%
import os
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def package_once(path):
    ...

X, y = load_breast_cancer(return_X_y=True)
print(X.shape)

# %% [markdown]
# Starter code for coding exercise 2.

# %%
from sklearn.datasets import load_breast_cancer

def build_app():
    ...

X, _ = load_breast_cancer(return_X_y=True)
print(len(X[0]))

# %% [markdown]
# Starter code for coding exercise 3.

# %%
PRICE_PER_GB_MONTH = {
    "Standard": 0.023,
    "Standard-IA": 0.0125,
    "Glacier Deep Archive": 0.00099,
}

def monthly_cost(footprint_gb, gb_read_back, gb_egress):
    ...

print(sorted(PRICE_PER_GB_MONTH))

# %% [markdown]
# Starter code for coding exercise 4.

# %%
ON_DEMAND_HOURLY = 0.0835
RESERVED_HOURLY = 0.05
HOURS = 730.0

def monthly_bills():
    ...

print(ON_DEMAND_HOURLY, RESERVED_HOURLY, HOURS)

# %% [markdown]
# Starter code for coding exercise 5.

# %%
def break_even(fixed, per_million):
    ...

print(round(0.0418 * 730 + 50 * 0.023, 2))

# %% [markdown]
# Starter code for coding exercise 6.

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def canary():
    ...

X, y = load_breast_cancer(return_X_y=True)
_, X_te, _, _ = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
print(len(X_te))

# %% [markdown]
# Starter code for coding exercise 7.

# %%
import boto3

def offline_report():
    ...

print(boto3.__name__)

# %% [markdown]
# Starter code for coding exercise 8.

# %%
policy = {
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::my-models-bucket/*",
        },
        {
            "Effect": "Deny",
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::my-models-bucket/prod/*",
        },
    ]
}

def allowed(policy, action, resource):
    ...

print([statement["Effect"] for statement in policy["Statement"]])

# %% [markdown]
# Starter code for coding exercise 9.

# %%
TABLE = {
    "object storage": {
        "AWS": "S3",
        "GCP": "Cloud Storage",
        "Azure": "Blob Storage",
    }
}

def translate(service_name, to_cloud):
    ...

print(sorted(TABLE))

# %% [markdown]
# Starter code for coding exercise 10.

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def mini():
    ...

X, y = load_breast_cancer(return_X_y=True)
_, X_te, _, _ = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
print(X.shape, len(X_te))

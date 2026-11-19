# Lab: deploy the one image to each cloud (ILLUSTRATIVE)

The cloud blocks need your own account, project, and region, and this book
does not run them, because they call the cloud providers. Check each command
against the provider's documentation before you run it, and check each
provider's pricing page for your region before you deploy, since prices change.

## The shared Dockerfile (see Chapter 28 for the build details)

```dockerfile
FROM python:3.13.18-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model.joblib app.py ./
# Cloud Run injects PORT (8080 unless you set another port). Container Apps and
# ECS Express Mode send traffic to the port you declare instead, so their deploy
# commands declare 8080, the default set here (30.10.2).
ENV PORT=8080
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
```

Build and tag it once. All three services run linux/amd64 images, Cloud Run and
Container Apps nothing else, and ECS Express Mode uses x86_64 unless you choose
ARM64 (30.10.2), so on an arm64 machine, an Apple silicon Mac for one, add the
platform flag, as the ECS guide tells such a reader to, or the build produces an
image of the machine's own architecture:

```bash
docker build -t breast-cancer-api:1.0 .
# On an arm64 machine (an Apple silicon Mac, for one), build for linux/amd64 instead:
docker build --platform linux/amd64 -t breast-cancer-api:1.0 .
```

## AWS: ECS Express Mode from a pushed image

(App Runner used to fill this role. As of November 2026 its page says that it
accepts no new customers from April 30, 2026, that existing services keep
running with no new features planned, and that AWS recommends ECS Express Mode
for containerized applications (Amazon Web Services, 2026u).)

```bash
# Push to Elastic Container Registry (the lines of 30.4.4 with the tag 1.0),
# then create an ECS Express Mode service. <acct> is your account number.
aws ecr create-repository --repository-name breast-cancer-api --region us-east-1
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <acct>.dkr.ecr.us-east-1.amazonaws.com
docker tag breast-cancer-api:1.0 \
  <acct>.dkr.ecr.us-east-1.amazonaws.com/breast-cancer-api:1.0
docker push <acct>.dkr.ecr.us-east-1.amazonaws.com/breast-cancer-api:1.0

# One-time setup: a task execution role (lets ECS pull the private ECR image
# and write logs) and an infrastructure role (lets ECS provision the load
# balancer, scaling, and networking), each with its ECS trust policy and the
# AWS managed policy of the Express Mode guide (30.10.2).
aws iam create-role --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
      "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

aws iam create-role --role-name ecsInfrastructureRoleForExpressServices \
  --assume-role-policy-document '{"Version": "2012-10-17", "Statement": [{
      "Sid": "AllowAccessInfrastructureForECSExpressServices", "Effect": "Allow",
      "Principal": {"Service": "ecs.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name ecsInfrastructureRoleForExpressServices \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices

# New roles can take about a minute to become usable: if the next command
# fails with an error about assuming a role right after you create them, wait
# and run it again.
aws ecs create-express-gateway-service \
  --service-name breast-cancer-api \
  --primary-container '{"image":"<acct>.dkr.ecr.us-east-1.amazonaws.com/breast-cancer-api:1.0",
      "containerPort":8080}' \
  --execution-role-arn arn:aws:iam::<acct>:role/ecsTaskExecutionRole \
  --infrastructure-role-arn arn:aws:iam::<acct>:role/ecsInfrastructureRoleForExpressServices \
  --health-check-path /health
# The execution role is what lets ECS pull the image from your private registry
# and write the container's logs.
# Returns a service URL like https://breast-cancer-api.ecs.us-east-1.on.aws
```

## GCP: Cloud Run, one command

```bash
# Push first with the tag and push lines of 30.5.3, tagging the local image
# breast-cancer-api:1.0 and pushing the tag 1.0 in place of v1.
gcloud run deploy breast-cancer-api \
  --image us-central1-docker.pkg.dev/my-ml-project-2026/ml-images/breast-cancer-api:1.0 \
  --region us-central1 \
  --allow-unauthenticated
# Returns a Service URL like https://breast-cancer-api-PROJECT_NUMBER.us-central1.run.app
```

## Azure: Container Apps, one command

```bash
# Push first with the login, tag, and push lines of 30.6.2, tagging the local
# image breast-cancer-api:1.0 and pushing the tag 1.0 in place of v1.
# <IDENTITY_ID> is the resource ID of a user-assigned identity made with
# az identity create; it needs the AcrPull role on mlregistry2026, an
# assignment the Azure CLI makes for you.
az containerapp create \
  --name breast-cancer-api \
  --resource-group ml-rg \
  --environment ml-env \
  --image mlregistry2026.azurecr.io/breast-cancer-api:1.0 \
  --target-port 8080 \
  --ingress external \
  --user-assigned <IDENTITY_ID> \
  --registry-identity <IDENTITY_ID> \
  --registry-server mlregistry2026.azurecr.io
# Returns an FQDN like breast-cancer-api.xxxx.eastus.azurecontainerapps.io
```

Same image, three public HTTPS endpoints. Test any of them with:

```bash
curl -s -X POST "$URL/predict" \
  -H "content-type: application/json" \
  -d '{"values": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
        0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.0064,
        0.049, 0.0537, 0.0159, 0.03, 0.0062, 25.38, 17.33, 184.6, 2019.0,
        0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}'
# -> {"probability_benign": 0.0, "label": 0}   (a malignant example; the app
#    rounds to 4 decimals, and this tumor's benign probability rounds to 0.0)
```

# Survivor-model predictor. (FASTAPI + DOCKER + K8S)

This project helps you learn Building and Deploying an ML Model using a simple and real-world use case: predicting whether a person is diabetic based on health metrics. We’ll go from:
- ✅ Model Training
- ✅ Building the Model locally
- ✅ API Deployment with FastAPI
- ✅ Dockerization
- ✅ Kubernetes Deployment

---
# PROBLEM STATEMENT:
Predicting if a person would survive the titanic based on;
- Age
- Pclass
- Sex

---

## 🚀 Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/clumumba/Survivor-model.git
cd first-mlops-project
```

### 2. Create Virtual Environment

```
python3 -m venv .mlops
source .mlops/bin/activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

## Train the Model

```
python train.py
```

## Run the API Locally

```
uvicorn main:app --reload
```

### Sample Input for /predict

```
{
  "SEX": 1,
  "Age": 45,
  "Pclass": 2
}
```

## Dockerize the API

### Build the Docker Image

```
docker build -t survivor-model .
```

### Run the Container

```
docker run -p 8000:8000 survivor-model
```

🙌 Credits

Created by `CLumumba`


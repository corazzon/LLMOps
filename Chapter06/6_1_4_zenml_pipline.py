from zenml.pipelines import pipeline
from zenml.steps import step

@step
def preprocess_data():
    print("Preprocessing data for LLM training or inference.")

@step
def deploy_model():
    print("Deploying the containerized LLM to Kubernetes.")

@pipeline
def llm_pipeline(preprocess_data, deploy_model):
    preprocess_data()
    deploy_model()

pipeline_instance = llm_pipeline(preprocess_data=preprocess_data(),

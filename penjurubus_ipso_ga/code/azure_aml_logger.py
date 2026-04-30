from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Trial
from azure.ai.ml.constants import AssetTypes
import json
import os

def get_aml_client():
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    credential = DefaultAzureCredential()
    subscription_id = os.environ["AML_SUBSCRIPTION_ID"]
    resource_group = os.environ["AML_RESOURCE_GROUP"]
    workspace = os.environ["AML_WORKSPACE_NAME"]
    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )

def log_ipsoga_run(experiment_name: str, run_name: str, metrics: dict, best_halte: list):
    ml_client = get_aml_client()
    run = ml_client.runs.create_or_update(
        job_name=run_name,
        display_name=run_name,
        experiment_name=experiment_name,
        description="PenjuruBus IPSO‑GA for halte location + route design",
    )

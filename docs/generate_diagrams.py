"""
AWS Architecture Diagrams for Deepfake Detection Workshop
- Smaller, cleaner diagrams
"""
from diagrams import Diagram, Cluster, Edge
from diagrams.aws.storage import S3
from diagrams.aws.ml import Sagemaker, SagemakerModel, SagemakerNotebook
from diagrams.aws.management import Cloudwatch
from diagrams.onprem.client import Users
from diagrams.programming.framework import React
import os

# Output directory
OUTPUT_DIR = "./images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Smaller graph attributes
graph_attr = {
    "fontsize": "14",
    "bgcolor": "white",
    "pad": "0.3",
    "dpi": "100",  # Lower DPI for smaller file size
    "size": "8,6",  # Limit size
}

node_attr = {
    "fontsize": "10",
    "width": "1.2",
    "height": "1.2",
}

edge_attr = {
    "fontsize": "9",
}


def create_overall_architecture():
    """전체 워크샵 아키텍처 (단순화)"""
    with Diagram(
        "Workshop Architecture",
        filename=f"{OUTPUT_DIR}/01_overall_architecture",
        show=False,
        direction="LR",
        graph_attr={**graph_attr, "size": "10,4"},
        node_attr=node_attr,
    ):
        user = Users("User")
        notebook = SagemakerNotebook("SageMaker\nStudio")
        s3 = S3("S3")
        training = Sagemaker("Training\nJobs")
        registry = SagemakerModel("Model\nRegistry")
        endpoint = Sagemaker("Endpoint")

        user >> notebook >> s3 >> training >> registry >> endpoint >> user


def create_finetuning_architecture():
    """Fine-tuning 파이프라인 (단순화된 버전)"""
    with Diagram(
        "SageMaker Training Pipeline",
        filename=f"{OUTPUT_DIR}/02_finetuning_architecture",
        show=False,
        direction="LR",
        graph_attr={**graph_attr, "size": "10,5"},
        node_attr=node_attr,
    ):
        s3_input = S3("S3\nInput")

        with Cluster("Training Jobs"):
            full = Sagemaker("Full")
            freeze = Sagemaker("Freeze")
            lora = Sagemaker("LoRA")

        s3_output = S3("S3\nOutput")
        registry = SagemakerModel("Model\nRegistry")

        s3_input >> full >> s3_output
        s3_input >> freeze >> s3_output
        s3_input >> lora >> s3_output
        s3_output >> registry


def create_deployment_architecture():
    """배포 아키텍처 (단순화)"""
    with Diagram(
        "Endpoint Deployment",
        filename=f"{OUTPUT_DIR}/03_deployment_architecture",
        show=False,
        direction="LR",
        graph_attr={**graph_attr, "size": "8,3"},
        node_attr=node_attr,
    ):
        user = Users("User")
        gradio = React("Gradio UI")
        endpoint = Sagemaker("Endpoint")
        model = SagemakerModel("Model")

        user >> Edge(label="image") >> gradio >> endpoint >> model
        model >> Edge(label="REAL/FAKE") >> gradio >> user


def create_transfer_learning():
    """Transfer Learning 개념도 (단순화)"""
    with Diagram(
        "Transfer Learning",
        filename=f"{OUTPUT_DIR}/05_transfer_learning",
        show=False,
        direction="LR",
        graph_attr={**graph_attr, "size": "8,3"},
        node_attr=node_attr,
    ):
        imagenet = S3("ImageNet\n1.4M")
        pretrained = SagemakerModel("Pretrained")
        kodf = S3("KoDF")
        finetuned = SagemakerModel("Fine-tuned")

        imagenet >> pretrained >> Edge(label="transfer") >> finetuned
        kodf >> finetuned


if __name__ == "__main__":
    print("Generating architecture diagrams...")

    create_overall_architecture()
    print("✅ 01_overall_architecture.png")

    create_finetuning_architecture()
    print("✅ 02_finetuning_architecture.png")

    create_deployment_architecture()
    print("✅ 03_deployment_architecture.png")

    create_transfer_learning()
    print("✅ 05_transfer_learning.png")

    print("\n🎉 Done! Check ./images/")

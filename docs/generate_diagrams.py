"""
AWS Architecture Diagrams for Deepfake Detection Workshop
"""
from diagrams import Diagram, Cluster, Edge
from diagrams.aws.storage import S3
from diagrams.aws.ml import Sagemaker, SagemakerModel, SagemakerNotebook
from diagrams.aws.management import Cloudwatch
from diagrams.aws.compute import EC2
from diagrams.aws.general import User
from diagrams.onprem.client import Users
from diagrams.programming.framework import React
from diagrams.custom import Custom
import os

# Output directory
OUTPUT_DIR = "./images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Graph attributes for better styling
graph_attr = {
    "fontsize": "20",
    "bgcolor": "white",
    "pad": "0.5",
    "splines": "spline",
}

node_attr = {
    "fontsize": "12",
}

edge_attr = {
    "fontsize": "10",
}

def create_overall_architecture():
    """전체 워크샵 아키텍처"""
    with Diagram(
        "Deepfake Detection Workshop Architecture",
        filename=f"{OUTPUT_DIR}/01_overall_architecture",
        show=False,
        direction="LR",
        graph_attr=graph_attr,
        node_attr=node_attr,
    ):
        user = Users("Workshop\nParticipants")

        with Cluster("SageMaker Studio"):
            notebook = SagemakerNotebook("Jupyter\nNotebook")

        with Cluster("Data Layer"):
            s3_data = S3("S3 Bucket\n(KoDF Data)")

        with Cluster("Training Layer"):
            training = Sagemaker("Training Jobs\n(GPU)")
            experiments = Sagemaker("Experiments\n(Tracking)")

        with Cluster("Model Management"):
            s3_model = S3("S3 Bucket\n(Models)")
            registry = SagemakerModel("Model\nRegistry")

        with Cluster("Inference Layer"):
            endpoint = Sagemaker("Real-time\nEndpoint")

        user >> notebook
        notebook >> s3_data
        notebook >> training
        training >> experiments
        training >> s3_model
        s3_model >> registry
        registry >> endpoint
        endpoint >> user


def create_finetuning_architecture():
    """Fine-tuning 파이프라인 아키텍처"""
    with Diagram(
        "SageMaker Training + Experiments",
        filename=f"{OUTPUT_DIR}/02_finetuning_architecture",
        show=False,
        direction="LR",
        graph_attr=graph_attr,
        node_attr=node_attr,
    ):
        s3_input = S3("S3 Input\n(Training Data)")

        with Cluster("SageMaker Training Jobs × 3"):
            full = Sagemaker("Full\nFine-tune")
            freeze = Sagemaker("Freeze\nLayers")
            lora = Sagemaker("LoRA\nAdapter")

        with Cluster("Outputs"):
            s3_output = S3("S3 Output\n(Model Artifacts)")
            registry = SagemakerModel("Model\nRegistry")
            logs = Cloudwatch("CloudWatch\nLogs")
            experiments = Sagemaker("Experiments\n(Tracking)")

        s3_input >> Edge(label="input data") >> full
        s3_input >> freeze
        s3_input >> lora

        full >> Edge(label="model artifact") >> s3_output
        freeze >> s3_output
        lora >> s3_output

        full >> registry
        freeze >> registry
        lora >> registry

        full >> Edge(style="dashed") >> logs
        freeze >> Edge(style="dashed") >> logs
        lora >> Edge(style="dashed") >> logs

        full >> Edge(style="dashed") >> experiments
        freeze >> Edge(style="dashed") >> experiments
        lora >> Edge(style="dashed") >> experiments


def create_deployment_architecture():
    """배포 아키텍처"""
    with Diagram(
        "SageMaker Endpoint Deployment",
        filename=f"{OUTPUT_DIR}/03_deployment_architecture",
        show=False,
        direction="LR",
        graph_attr=graph_attr,
        node_attr=node_attr,
    ):
        user = Users("Users")

        with Cluster("Demo Interface"):
            gradio = React("Gradio\nWeb UI")

        with Cluster("SageMaker"):
            endpoint = Sagemaker("Real-time\nEndpoint")

            with Cluster("ml.g4dn.xlarge"):
                model = SagemakerModel("PyTorch\nModel")

        with Cluster("Model Storage"):
            s3_model = S3("S3\n(model.tar.gz)")

        user >> Edge(label="Upload Image") >> gradio
        gradio >> Edge(label="invoke_endpoint") >> endpoint
        endpoint >> model
        s3_model >> Edge(label="load model") >> model
        model >> Edge(label="REAL/FAKE") >> gradio
        gradio >> Edge(label="Result") >> user


def create_data_flow():
    """데이터 흐름도"""
    with Diagram(
        "Workshop Data Flow",
        filename=f"{OUTPUT_DIR}/04_data_flow",
        show=False,
        direction="TB",
        graph_attr=graph_attr,
        node_attr=node_attr,
    ):
        with Cluster("Step 1: Data Preparation"):
            s3_public = S3("Public S3\n(Workshop Data)")
            s3_private = S3("Your S3\n(Copy)")

        with Cluster("Step 2: Before Evaluation"):
            pretrained = SagemakerModel("Pretrained\nModel (FF++)")
            eval1 = Sagemaker("Evaluate\n~70%")

        with Cluster("Step 3: Fine-tuning"):
            training = Sagemaker("Training × 3\n(Full/Freeze/LoRA)")

        with Cluster("Step 4: After Evaluation"):
            finetuned = SagemakerModel("Fine-tuned\nModel (KoDF)")
            eval2 = Sagemaker("Evaluate\n~90%")

        with Cluster("Step 5-6: Compare & Deploy"):
            compare = Sagemaker("Performance\nComparison")
            endpoint = Sagemaker("Endpoint\nDeployment")

        s3_public >> s3_private
        s3_private >> pretrained >> eval1
        s3_private >> training
        training >> finetuned >> eval2
        eval1 >> compare
        eval2 >> compare
        compare >> endpoint


def create_transfer_learning():
    """Transfer Learning 개념도"""
    with Diagram(
        "Transfer Learning Concept",
        filename=f"{OUTPUT_DIR}/05_transfer_learning",
        show=False,
        direction="LR",
        graph_attr=graph_attr,
        node_attr=node_attr,
    ):
        with Cluster("Source Domain"):
            imagenet = S3("ImageNet\n(1.4M images)")
            pretrained = SagemakerModel("Pretrained\nEfficientNet")

        with Cluster("Knowledge Transfer"):
            transfer = Sagemaker("Fine-tuning")

        with Cluster("Target Domain"):
            kodf = S3("KoDF\n(Korean Faces)")
            finetuned = SagemakerModel("Fine-tuned\nModel")

        imagenet >> pretrained
        pretrained >> Edge(label="Transfer\nWeights") >> transfer
        kodf >> transfer
        transfer >> finetuned


if __name__ == "__main__":
    print("Generating architecture diagrams...")

    create_overall_architecture()
    print("✅ 01_overall_architecture.png")

    create_finetuning_architecture()
    print("✅ 02_finetuning_architecture.png")

    create_deployment_architecture()
    print("✅ 03_deployment_architecture.png")

    create_data_flow()
    print("✅ 04_data_flow.png")

    create_transfer_learning()
    print("✅ 05_transfer_learning.png")

    print("\n🎉 All diagrams generated in ./images/")

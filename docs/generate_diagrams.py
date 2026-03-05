"""
AWS Architecture Diagrams - Larger fonts
"""
from diagrams import Diagram, Cluster, Edge
from diagrams.aws.storage import S3
from diagrams.aws.ml import Sagemaker, SagemakerModel, SagemakerNotebook
from diagrams.onprem.client import Users
from diagrams.programming.framework import React
import os

OUTPUT_DIR = "./images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fix: smaller icons, larger fonts
graph_attr = {
    "fontsize": "16",
    "bgcolor": "white",
    "pad": "0.3",
    "dpi": "120",
    "ranksep": "0.8",
    "nodesep": "0.6",
}

node_attr = {
    "fontsize": "11",
    "width": "1.2",
    "height": "1.2",
}

edge_attr = {
    "fontsize": "12",
}


def create_overall_architecture():
    """전체 워크샵 아키텍처"""
    with Diagram(
        "Workshop Architecture",
        filename=f"{OUTPUT_DIR}/01_overall_architecture",
        show=False,
        direction="LR",
        graph_attr={**graph_attr, "size": "14,6"},
        node_attr=node_attr,
        edge_attr=edge_attr,
    ):
        user = Users("User")
        notebook = SagemakerNotebook("SageMaker\nStudio")
        s3 = S3("S3\nBucket")
        training = Sagemaker("Training\nJobs")
        registry = SagemakerModel("Model\nRegistry")
        endpoint = Sagemaker("Endpoint")

        user >> notebook >> s3 >> training >> registry >> endpoint >> user


def create_finetuning_architecture():
    """Fine-tuning 파이프라인"""
    with Diagram(
        "SageMaker Training Pipeline",
        filename=f"{OUTPUT_DIR}/02_finetuning_architecture",
        show=False,
        direction="LR",
        graph_attr={**graph_attr, "size": "14,8"},
        node_attr=node_attr,
        edge_attr=edge_attr,
    ):
        s3_input = S3("S3 Input\n(Training Data)")

        with Cluster("Training Jobs"):
            full = Sagemaker("Full\nFine-tuning")
            freeze = Sagemaker("Layer\nFreezing")
            lora = Sagemaker("LoRA")

        s3_output = S3("S3 Output\n(Models)")
        registry = SagemakerModel("Model\nRegistry")

        s3_input >> full >> s3_output
        s3_input >> freeze >> s3_output
        s3_input >> lora >> s3_output
        s3_output >> registry


def create_deployment_architecture():
    """배포 아키텍처"""
    with Diagram(
        "Endpoint Deployment",
        filename=f"{OUTPUT_DIR}/03_deployment_architecture",
        show=False,
        direction="LR",
        graph_attr={**graph_attr, "size": "12,5"},
        node_attr=node_attr,
        edge_attr=edge_attr,
    ):
        user = Users("User")
        gradio = React("Gradio\nWeb UI")
        endpoint = Sagemaker("SageMaker\nEndpoint")
        model = SagemakerModel("PyTorch\nModel")

        user >> Edge(label="image") >> gradio >> endpoint >> model
        model >> Edge(label="REAL/FAKE") >> gradio >> user


def create_transfer_learning():
    """Transfer Learning 개념도"""
    with Diagram(
        "Transfer Learning",
        filename=f"{OUTPUT_DIR}/05_transfer_learning",
        show=False,
        direction="LR",
        graph_attr={**graph_attr, "size": "12,5"},
        node_attr=node_attr,
        edge_attr=edge_attr,
    ):
        imagenet = S3("ImageNet\n(1.4M images)")
        pretrained = SagemakerModel("Pretrained\nModel")
        kodf = S3("Target\nDataset")
        finetuned = SagemakerModel("Fine-tuned\nModel")

        imagenet >> pretrained >> Edge(label="transfer weights") >> finetuned
        kodf >> finetuned


def create_video_detection_pipeline():
    """숏폼 영상 딥페이크 탐지 파이프라인"""
    from diagrams.onprem.compute import Server

    with Diagram(
        "Short-form Video Deepfake Detection Pipeline",
        filename=f"{OUTPUT_DIR}/00_video_detection_pipeline",
        show=False,
        direction="LR",
        graph_attr={**graph_attr, "size": "18,6"},
        node_attr=node_attr,
        edge_attr=edge_attr,
    ):
        user = Users("User\nUpload")

        with Cluster("Video Processing"):
            video = S3("Short-form\nVideo")
            frames = Server("Frame\nExtraction\n(3fps)")
            face = Server("Face\nDetection\n(MTCNN)")

        with Cluster("Fine-tuned Model (This Workshop)"):
            cnn = SagemakerModel("CNN\nClassifier")

        with Cluster("Result Aggregation"):
            aggregate = Server("Voting\n(Majority)")
            result = Users("REAL/FAKE")

        user >> video >> frames >> face >> cnn >> aggregate >> result


def create_production_architecture():
    """프로덕션 아키텍처 (비동기 추론)"""
    from diagrams.aws.integration import SNS, SQS
    from diagrams.aws.compute import Lambda

    with Diagram(
        "Production Architecture (Async Inference)",
        filename=f"{OUTPUT_DIR}/06_production_architecture",
        show=False,
        direction="LR",
        graph_attr={**graph_attr, "size": "16,7"},
        node_attr=node_attr,
        edge_attr=edge_attr,
    ):
        user = Users("Game\nPlatform")

        with Cluster("Request"):
            s3_input = S3("S3 Input\n(Video)")

        with Cluster("SageMaker Async Inference"):
            queue = SQS("Queue")
            endpoint = Sagemaker("Async\nEndpoint")

        with Cluster("Response"):
            s3_output = S3("S3 Output\n(Result)")
            sns = SNS("SNS\nCallback")

        user >> s3_input >> queue >> endpoint >> s3_output >> sns >> user


if __name__ == "__main__":
    print("Generating architecture diagrams...")
    create_video_detection_pipeline()
    print("✅ 00_video_detection_pipeline.png")
    create_overall_architecture()
    print("✅ 01_overall_architecture.png")
    create_finetuning_architecture()
    print("✅ 02_finetuning_architecture.png")
    create_deployment_architecture()
    print("✅ 03_deployment_architecture.png")
    create_transfer_learning()
    print("✅ 05_transfer_learning.png")
    create_production_architecture()
    print("✅ 06_production_architecture.png")
    print("\n🎉 Done!")

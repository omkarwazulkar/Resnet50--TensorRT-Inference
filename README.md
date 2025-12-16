# Resnet50--TensorRT-Inference
This project performs high-performance inference with ResNet-50 using TensorRT. It starts by obtaining a pre-trained ResNet-50 model, uses NVIDIA’s TensorRT Docker image from NGC, converts the model from PyTorch to ONNX, then builds a TensorRT engine, enabling extremely fast &amp; efficient inference with low latency &amp; high throughput on NVIDIA GPUs.

### **Step 1: Set Up SageMaker Notebook Environment**

- Launch an **Amazon SageMaker Notebook Instance**
- Use the following configuration:
  - **Instance type:** `g5.2xlarge`
  - **Storage:** 20 GB
- Attach an **IAM role** with permissions for:
  - **Amazon ECR**
  - **Amazon S3**
  - **Amazon SageMaker**

This notebook instance acts as the **central environment** for model conversion, container image builds, and deployment orchestration.

### **Step 2: Pull NVIDIA TensorRT**

- Authenticate with **NVIDIA NGC** (if required)
- Pull the official NVIDIA base images:
  - **TensorRT** image for ONNX → TensorRT engine builds

These base images are used for **model optimization** and **GPU-accelerated inference deployment**.

### **High-Performance ResNet-50 Inference with TensorRT**

This project performs **high-performance inference** with **ResNet-50** using **NVIDIA TensorRT**, following an optimized model conversion and deployment pipeline.

#### **Workflow Explanation**

- **Obtain a Pre-trained ResNet-50 Model**
  - A pre-trained ResNet-50 model is loaded from PyTorch (TorchVision).
  - This provides a strong baseline model without requiring training from scratch.

- **Use NVIDIA TensorRT Docker Image (NGC)**
  - NVIDIA’s official TensorRT Docker image is pulled from **NVIDIA NGC**.
  - The image includes **CUDA, cuDNN, TensorRT, and ONNX support**, ensuring a GPU-optimized environment.

- **Convert PyTorch Model to ONNX**
  - The ResNet-50 model is exported from PyTorch to **ONNX** format.
  - ONNX serves as a **framework-agnostic intermediate representation** for optimization and deployment.

- **Build TensorRT Engine**
  - The ONNX model is compiled into a **TensorRT engine**.
  - TensorRT applies optimizations such as:
    - Layer fusion
    - Kernel auto-tuning
    - Reduced precision (FP16 / INT8)
  - The resulting engine is **GPU-specific and highly optimized**.

- **Run GPU-Accelerated Inference**
  - Inference is performed using the TensorRT engine instead of the original PyTorch model.
  - This enables:
    - **Low latency**
    - **High throughput**
    - **Efficient GPU memory usage**

#### **Why TensorRT Improves Performance**

- Removes framework-level overhead
- Uses optimized GPU kernels
- Exploits reduced-precision computation
- Maximizes parallel execution on NVIDIA GPUs



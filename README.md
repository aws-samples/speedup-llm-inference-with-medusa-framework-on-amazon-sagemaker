# Medusa on Amazon Sagemaker AI

This repository supports setting up and running Medusa, a technique introduced in the ["Medusa: Simple LLM Inference 
Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) paper, on [Amazon SageMaker AI](https://aws.amazon.com/sagemaker-ai/).

This repository is a modified version of the original [How to Fine-Tune LLMs in 2024 on Amazon SageMaker](https://github.com/philschmid/llm-sagemaker-sample/blob/main/notebooks/train-evalaute-llms-2024-trl.ipynb) example by [Philipp Schmid](https://www.philschmid.de/philipp-schmid). We added a simplified Medusa training code, adapted from the original [Medusa framework repository](https://github.com/FasterDecoding/Medusa).  We use a dataset called [sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context), which contains samples of natural language instructions, schema definitions and the corresponding SQL query.

For more details and step-by-step instructions please refer to accompanying blog post: [Achieve ~2x speed-up in LLM inference with Medusa-1 on Amazon SageMaker AI](https://aws.amazon.com/blogs/machine-learning/achieve-2x-speed-up-in-llm-inference-with-medusa-1-on-amazon-sagemaker-ai/)

## Why use the Medusa framework on Amazon SageMaker AI?

Large Language Models generate text in a sequential manner, with each new token conditional on the previous ones.
This process can be slow, and different techniques have been suggested to address these issues one of which is Medusa framework. 
The original paper authors report inference token generation speed-ups of up to 2x-3.6x depending on the framework version.

Amazon SageMaker AI provides a fully managed machine learning service that makes it easy to build, train, and deploy 
high-quality machine learning (ML) models. Running Medusa framework on Amazon SageMaker AI allows you to quickly scale
up your Medusa model training process as the service handles heavy lifting of infrastructure provisioning for you. 
Furthermore, you can leverage Amazon SageMaker AI real-time endpoints with [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) container to run
scalably your inference server with Medusa inference speed-ups.

## Getting started

In order to follow the instructions in this repository you will need to setup:
- AWS account with [AWS Identity and Access Management (IAM) role](https://aws.amazon.com/iam/) with correct set of permissions
- Your development environment with the right permissions to access AWS account. Alternatively, use Amazon SageMaker Studio in your AWS account)

For more details please refer to the Prerequisites section of accompanying blog post: [Achieve ~2x speed-up in LLM inference with Medusa-1 on Amazon SageMaker AI](https://aws.amazon.com/blogs/machine-learning/achieve-2x-speed-up-in-llm-inference-with-medusa-1-on-amazon-sagemaker-ai/)

## Repository structure

You can find data processing, fine-tuning, and evaluation code in `medusa_1_train.ipynb` notebook. In the `train` folder you can find the scripts that are run inside the SageMaker AI Training job such as fine tuning the LLM and training Medusa heads.

We cover the following steps in `medusa_1_train.ipynb` notebook: 
1. Load and prepare the dataset
2. Fine-tune an LLM using a SageMaker AI training job
3. Train Medusa heads on top of a frozen fine-tuned LLM using a SageMaker AI training job
4. Deploy the fine-tuned LLM with Medusa heads on a SageMaker AI endpoint
5. Demonstrate LLM inference speedup: We compare average latencies between fine-tuned LLM and the fine-tuned LLM with Medusa heads.

## Security
See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License
This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.

## Authors and acknowledgment

* Daniel Zagyva
* Aleksandra Dokic
* Laurens van der Maas
* Manos Stergiadis
* Moran Beladev
* Ilya Gusev

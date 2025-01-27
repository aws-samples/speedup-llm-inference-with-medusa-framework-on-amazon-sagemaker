# Medusa on Amazon Sagemaker

This repository supports setting up and running Medusa, a technique introduced in the ["Medusa: Simple LLM Inference 
Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) paper, on [Amazon SageMaker](https://aws.amazon.com/pm/sagemaker/?gclid=Cj0KCQjwmOm3BhC8ARIsAOSbapVEm--Q2sgQ7QFKgdo5epDmZZ0g8uYJ1sFPVbSQpbdizEkDTP5hVB0aAjXoEALw_wcB&trk=3ea5c9d1-0497-4ab3-92e6-c583f43ac2f9&sc_channel=ps&ef_id=Cj0KCQjwmOm3BhC8ARIsAOSbapVEm--Q2sgQ7QFKgdo5epDmZZ0g8uYJ1sFPVbSQpbdizEkDTP5hVB0aAjXoEALw_wcB:G:s&s_kwcid=AL!4422!3!645186192649!e!!g!!amazon%20sagemaker!19571721771!146073031580).

This repository is a modified version of the original [How to Fine-Tune LLMs in 2024 on Amazon SageMaker](https://github.com/philschmid/llm-sagemaker-sample/blob/main/notebooks/train-evalaute-llms-2024-trl.ipynb) example by [Philipp Schmid](https://www.philschmid.de/philipp-schmid). We added a simplified Medusa training code, adapted from the original [Medusa framework repository](https://github.com/FasterDecoding/Medusa).  We use a dataset called [sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context), which contains samples of natural language instructions, schema definitions and the corresponding SQL query.

We cover the following steps in this repository: 
1. Load and prepare the dataset
2. Fine-tune an LLM using SageMaker Training Job
3. Train Medusa heads on top of frozen fine-tuned LLM using SageMaker Training Job
4. Deploy the fine-tuned LLM with Medusa heads on SageMaker Endpoint
5. Demonstrate LLM inference speedup: We compare average latencies between fine-tuned LLM and the fine-tuned LLM with Medusa heads.


## Why using Medusa framework on Amazon SageMaker?

Large Language Models generate text in a sequential manner, with each new token conditional on the previous ones.
This process can be slow, and different techniques have been suggested to address these issues one of which is Medusa framework. 
The original paper authors report inference token generation speed-ups of up to 2x-3.6x depending on the framework version.

Amazon SageMaker provides a fully managed machine learning service that makes it easy to build, train, and deploy 
high-quality machine learning (ML) models. Running Medusa framework on Amazon SageMaker would allow you to quickly scale
up your Medusa model training process as the service handles heavy lifting of infrastructure provisioning for you. 
Furthermore, you can leverage Amazon SageMaker real-time endpoints with [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) container to run
scalably your inference server with Medusa inference speed-ups. 

For more details on and step-by-step instructions please refer to accompanying blog post [TODO: add link].

## Getting started

In order to follow the instructions in this repository you will need to setup:
- AWS account with [AWS Identity and Access Management (IAM) role](https://aws.amazon.com/iam/) with correct set of permissions
- Your development environment with the right permissions to access AWS account. Alternative, Amazon SageMaker Studio domain in your AWS account)

For more details please refer to [TODO: add blog post link to the right chapter].

## Repository structure

You can find data processing, fine-tuning, and evaluation code in `medusa_train.ipynb` notebook. In the `train` folder you can find the scripts that are run inside the SageMaker Training job such as fine tuning the LLM and training Medusa heads.

## Security
See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License
This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.

## Authors and acknowledgment

Daniel Zagyva

Aleksandra Dokic

Laurens van der Maas

Manos Stergiadis

Moran Beladev

Ilya Gusev

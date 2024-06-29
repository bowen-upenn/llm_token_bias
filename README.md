## This is the official implementation of the paper ["A Peek into Token Bias: Large Language Models Are Not Yet Genuine Reasoners"](https://arxiv.org/pdf/2406.11050) in PyTorch.

💜 **This README also provides step-by-step guidance on setting up a variety of the most popular LLMs APIs.**

Large language models (LLMs) have achieved remarkable progress in understanding and generating human-like text, but there is ongoing debate about whether LLMs possess **genuine reasoning capabilities**. This work reconceptualizes the evaluation of LLM's reasoning capabilities into a general and rigorous testing framework with statistical guarantee. 

We say that an LLM is subject to **token bias** in a reasoning task if **systematic** changes to some or all tokens in the task descriptions - while keeping the underlying logic intact - allow us to **predict** the direction of the shift in the model’s output. A strong token bias suggests that LLM is relying on superficial patterns in the input rather than truly understanding the underlying reasoning task, leading to brittle performance that fails to generalize well. 

Comprehensive experiments on both commercial and open-sourced LLMs on large-scale synthetic datasets uncover a critical insight: **It is the token bias that contributes the most to performance improvements in reasoning tasks, if any, rather than genuine advances in reasoning capabilities.**

We explore several well-known logical fallacy problems from the cognitive science literature as a clean experimental playground. 
Following is an example of the classical **Linda Problem**.
> Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations. Which is more probable?
>
> (a) Linda is a bank teller. :ok_woman:
> 
> (b) Linda is a bank teller and is active in the feminist movement. :sassy_woman:

Experiments in behavioral psychology reveal that people typically believed the second option was more likely than the first, but this contradicts the basic **probability rule of conjunction**. Advanced LLMs like GPT-4 can typically recognize this fallacy well since it is a classical problem that appears frequently in cognitive science literature. However, altering seemingly irrelevant tokens, like the name "Linda" in the problem statement, while maintaining the same logical structure can surprisingly confuse most LLMs, leading to our concern that LLMs are not yet genuine reasoners. Please see detailed token perturbations in our [paper](https://arxiv.org/pdf/2406.11050). 


## Dependencies
Please check [requirements.txt](requirements.txt). You can run the following commands to create a virtual environment and install all the requirements:
    
    python -m venv myenv
    source myenv/bin/activate
    pip install -r requirements.txt

## Citation
This bunny 🐰 will be happy if you could cite our work. Thank you!

    @article{jiang2024peek,
      title={A Peek into Token Bias: Large Language Models Are Not Yet Genuine Reasoners},
      author={Jiang, Bowen and Xie, Yangxinyu and Hao, Zhuoqun and Wang, Xiaomeng and Mallick, Tanwi and Su, Weijie J and Taylor, Camillo J and Roth, Dan},
      journal={arXiv preprint arXiv:2406.11050},
      year={2024}
    }

## Dataset
We provide our synthetic dataset under [data/](data/), which contains a comprehensive set of logical fallacies like the Linda Problem. The dataset file is in JSON format, and each item is a dictionary containing ```question_id```, ```question```, ```target_answer```, and ```incorrect_answer```. You can also generate more synthetic data on the fly.

## LLM Setups
:heart: Always set up **OpenAI ChatGPT** models. Please follow its [Developer quickstart](https://platform.openai.com/docs/quickstart?context=python) to set up your OpenAI API, create a new [api_tokens/openai_key.txt](api_tokens/openai_key.txt) file, and copy and paste your [API key](https://platform.openai.com/api-keys) into it.

:orange_heart: To use **Google Gemini** models with an API for inference, follow instructions on [Google Vertex AI](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemini-pro) about the ```Try Gemini 1.0 Pro (Python)``` section. Note that your school's Gmail account may not allow you to make payments.
- Step 1: According to their instructions, you need to first [install the Vertex AI client libraries](https://cloud.google.com/vertex-ai/docs/start/client-libraries#before_you_begin) to create a project with a project ID, enable Vertex AI API,  create a service account, and generate your account key. You don't need to set the environment variable ```GOOGLE_APPLICATION_CREDENTIALS``` since we have already done that for you in our codes [query_llm.py](query_llm.py).
- Step 2: Install or update the [Vertex AI SDK for Python](https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk?_gl=1*1qxspr1*_ga*OTIxODQ4MjQzLjE3MTQwNzMwNjI.*_ga_WH2QY8WWF5*MTcxNDE1NDc1Ni4yLjEuMTcxNDE2MDMzMi4wLjAuMA..&_ga=2.54138400.-921848243.1714073062).
- Step 3: Authenticate to Vertex AI and set up [Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc?&_ga=2.268814513.-1366866501.1708968072#local-dev).
  - Follow the ```Local development environment - Provide user credentials for your Google Account``` section to install and initialize the [gcloud CLI](https://cloud.google.com/sdk/docs/install). This step will download a folder ```google-cloud-sdk``` to your project's top directory.
  - After installation, run
    
        gcloud init
    
     to initialize the gcloud CLI. You will be able to choose your account and project ID. Create a new [api_tokens/gemini_project_id.txt](api_tokens/gemini_project_id.txt) file, and copy and paste your project ID into it.
  - To create your credential file, run
    
        gcloud auth application-default login
    
     You will see a prompt like ```Credentials saved to file: [/path/to/your/home/.config/gcloud/application_default_credentials.json]```.
  - Because of the path of the credential file we set in our [config.yaml](config.yaml), run
    
        mv /path/to/your/home/.config/gcloud/application_default_credentials.json google-cloud-sdk/google_gemini_credential.json

:yellow_heart: To use **Meta Llama** models with an API for inference, follow instructons on [Replicate Run Llama 3 with an API](https://replicate.com/blog/run-llama-3-with-an-api?utm_source=project&utm_campaign=llama2ai) about the ```Running Llama 3 with Python``` section to set up your [API tokens](https://replicate.com/account/api-tokens), create a new [api_tokens/llama_key.txt](api_tokens/llama_key.txt) file, and copy and paste your tokens into it.

:green_heart: To use **Anthropic Claude** models with an API for inference, follow its [Quickstart Guide](https://docs.anthropic.com/claude/docs/quickstart-guide) to install the Anthropic Python SDK, set up an account with API access, get your [API key](https://console.anthropic.com/settings/keys), create a new [api_tokens/claude_key.txt](api_tokens/claude_key.txt) file, and copy and paste your key into it. You don't need to set the environment variable ```ANTHROPIC_API_KEY```.

:blue_heart: To use **Mistral** models with an API for inference, follow its [Quickstart](https://docs.mistral.ai/getting-started/quickstart/) to install the mistralai library, set up an account with API access, get your [API key]([https://console.anthropic.com/settings/keys](https://console.mistral.ai/api-keys/), create a new [api_tokens/mistral_key.txt](api_tokens/mistral_key.txt) file, and copy and paste your key into it. You don't need to set the environment variable ```MISTRAL_API_KEY```.

## Quick Start
We allow command-line argparser for the following arguments: 

- ```--model``` to select the LLM for inference. Last updated on 04-25-2024, but our codes should be compatible with any more recent model names. 
  
  - **OpenAI ChatGPT family.** Check [OpenAI's continuous model upgrades](https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4).
    - ```gpt3.5``` or equivalently ```gpt-3.5-turbo```, ```gpt-3.5-turbo-0125```
    - ```gpt-3.5-turbo-1106```
    - ```gpt-3.5-turbo-0613```
    - ```gpt4``` or equivalently  ```gpt-4-turbo```, ```gpt-4-turbo-2024-04-09```
    - ```gpt-4-0125-preview```
    - ```gpt-4-1106-preview```
    - ```gpt-4-0613```
  - **Google Gemini family.** Check [Gemini model versions and lifecycle](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning#auto-updated-version).
    - ```gemini``` or equivalently ```gemini-1.0-pro```, ```gemini-1.0-pro-002```
    - ```gemini-1.0-pro-001```
    - ```gemini-1.5-pro-preview-0409```
  - **Meta Llama family**. Check [Choosing which model to use Llama-3](https://replicate.com/blog/run-llama-3-with-an-api?utm_source=project&utm_campaign=llama2ai#choosing-which-model-to-use) and [Llama-2](https://replicate.com/blog/run-llama-2-with-an-api#choosing-which-model-to-use).
    - ```llama``` or equivalently ```llama3-70b```, ```meta-llama-3-70b-instruct```
    - ```llama3-8b``` or equivalently ```meta-llama-3-8b-instruct```
    - ```llama-2-70b-chat```
    - ```llama-2-13b-chat```
    - ```llama-2-7b-chat```
  - **Anthropic Claude family**. Check [Models overview](https://docs.anthropic.com/claude/docs/models-overview).
    - ```claude``` or equivalently ```claude-3-opus-20240229```
    - ```claude-3-sonnet-20240229```
    - ```claude-3-haiku-20240307```
  - **Mistral family**. Check [API versioning](https://docs.mistral.ai/getting-started/models/#api-versioning).
    - ```mistral``` or equivalently ```mistral-large-latest```, ```mistral-large-2402```
    - ```mistral-medium-latest``` or equivalently ```mistral-medium-2312```
    - ```mistral-small-latest``` or equivalently ```mistral-small-2402```
    - ```open-mixtral-8x22b``` or equivalently ```open-mixtral-8x22b-2404```
    - ```open-mixtral-8x7b``` or equivalently ```mistral-small-2312```
    - ```open-mistral-7b``` or equivalently ```mistral-tiny-2312```

- ```--task``` to either generate synthetic datasets: ```data``` or evaluate the LLM's ability to answer the questions: ```inference```.

- ```--fallacy``` to select the type of logical fallacy. We only support ```linda``` at this moment for the Linda Problem and its variants.

- ```--verbose``` to print detailed data information and model responses during the inference.

- ***\[For Data Gen Only\]*** ```--gen_mode``` to select the mode of generating synthetic dataset when ```task``` is ```data```. Options are ```baseline```: simple in-context learning with limited instructions, ```control```: step-by-step guidance to generate both gold samples and random samples with irrelevant info.

- ***\[For Data Gen Only\]*** ```--variant``` to select the variant of the Linda problems, such as the default ```original```, ```variant_one```, ```variant_two```, ..., ```variant_six```. Detailed information about each variant can be found in the ```def linda_problem()``` function in [prompts.py](prompts.py).

- ***\[For Data Gen Only\]*** ```--conn``` to select the logical connecting word, such as ```because```, ```sothat```, or ```to``` when using ```variant_one``` or ```variant_two``` to generate new data.

- ***\[For Data Gen Only\]*** ```--n``` to set the number of synthetic data problems to generate.

- ***\[For Inference Only\]*** ```--data_file``` to set the data file path for inference.

- ***\[For Inference Only\]*** In progress ```--multi_agent``` to enable a multi-agent system mimicking a debating scenario among multiple LLMs for better performance.

- ***\[For Inference Only\]*** ```--eval_mode``` to set the evaluation mode for the model to answer questions. Options are 
  - ```baseline``` for directly prompting
  - ```zs_cot``` for zero-shot chain-of-thought (CoT) prompting
  - ```os``` for one-shot in-context learning (ICL) prompting with the original Linda Problem (default)
  - ```os_cot``` for one-shot ICL plus COT prompting
  - ```os_bob``` for one-shot ICL prompting but with a rephrased Bob Problem
  - ```os_bob_cot``` for one-shot ICL prompting plus COT but with a rephrased Bob Problem
  - ```os_incorrect``` for one-shot ICL but with an incorrect answer
  - ```os_incorrect_cot``` for one-shot ICL plus COT but with an incorrect answer
  - ```fs``` for few-shot ICL prompting
  - ```fs_cot``` for few-shot ICL plus COT prompting
  - ```weak_control_zs_cot``` for weakly controlled zero-shot CoT prompting, leaking the hint that it is a Linda Problem but without detailed explanations
  - ```weak_control_os_cot``` for weakly controlled one-shot CoT prompting, leaking the hint that it is a Linda Problem but without detailed explanations
  - ```control_zs_cot``` for controlled zero-shot CoT prompting, leaking the hint that it is a Linda Problem with detailed and carefully-curated explanations
  - ```control_os_cot``` for controlled one-shot CoT prompting, leaking the hint that it is a Linda Problem with detailed and carefully-curated explanations


**To start the inference code**

    python main.py --model gpt3.5 --task inference --eval_mode os_cot --data_file synthetic_dataset_linda_original_gold.json --verbose

in the command line and adjust the ``eval_mode`` and ``data_file``. 

You can also run 

    python main.py --model gpt3.5 --task data --fallacy linda --gen_mode control --variant original --n 100 --verbose

to generate synthetic datasets for the Linda Problem. All the other hyper-parameters can be set at [config.yaml](config.yaml).

To run inference on a data file with multiple prompting methods in parallel efficiently, modify the number of GPU devices available, the data file name to be evaluated ```--data_file```, the LLM model ```--model```, and the list of prompting methods in ``run.sh``. 
You can then run

    bash run.sh

All results and final accuracies will be automatically saved to the [outputs/](outputs/) directory.

  
python main.py --model gpt3.5 --task inference --eval_mode baseline --data_file synthetic_dataset_linda_variant_six_gold.json --verbose

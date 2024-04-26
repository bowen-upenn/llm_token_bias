## This is the official implementation of the paper "Large Language Models Are Not Yet Good Probabilistic Thinkers" in Pytorch.

The big question in this work is to identify when LLMs might fail and exhibit **stereotypes** in their decision-making and judgment at **logical fallacies**, and what strategies could help 
    
- ***An interesting intersection between LLMs and Psychology.***

We investigate "representative thinking" in this project, i.e., the problem of **learning from “experience”**. Specifically, we examine the [Conjunction Fallacy](https://en.wikipedia.org/wiki/Conjunction_fallacy), also known as the Linda Problem, that demonstrates human cognitive bias.
We refer readers interested in this topic to the books named [The Undoing Project](https://en.wikipedia.org/wiki/The_Undoing_Project) and [Thinking, Fast and Slow](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) for further details.

Following is an example of the classical **Linda Problem**.
> Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations. Which is more probable?
>
> (1) Linda is a bank teller. :purple_heart:
> 
> (2) Linda is a bank teller and is active in the feminist movement. :blue_heart:

Experiments in behavioral psychology reveal that people typically believed the second option was more likely than the first, but this contradicts the basic **probability rule of conjunction**. 
Similarly, LLMs may be misled by irrelevant context information in the problem statement, dive into unnecessary background knowledge with stereotypes, and find it hard to extract the underlying probabilistic model from the question. We doubt if alignment from human feedback has solved this problem yet and believe that LLMs are not yet good probabilistic thinkers.

**Key Motivations:**
 - The original Linda Problem is famous enough, so it is reasonable to believe that it has been included in the **training dataset** of most large language models. Creating a generalized evaluation dataset at **a much larger scale** becomes urgent. Most related works evaluate the model's performance on a very small dataset.
 - Even the increasingly large and powerful large language models might fail to recognize rephrased Linda problems in **different domains**, even if they still involve a conjunction fallacy. This raises cautionary implications for LLM being used in critical decision making.
 - We find In-Context Learning (ICL) powerful if the one-shot examplar is the original Linda Problem, even if the follow-up problem is rephrased in a different domain. However, if we simply rename 'Linda' to 'Bob' the one-shot examplar, ICL would suprisingly fail. We suspect that LLMs might **overfit to the name 'Linda'** in the original Linda Problem during their fine-tuning, **without understanding the actual reasonings**.
 - This phenomenon also calls for a **large-scale** synthetic dataset that covers a comprehensive set of Linda Problems in **diverse domains**, together with more carefully **controlled experiments** and ablation studies aligned with psychology and seeded from **real-world data sources**, to more thoroughly understand why Linda Problems are tricky and examine LLM's ability as a probabilistic thinker.
 - To solve Linda Problems, humans must recognize the conjunction fallacy that lies beneath the irrelevant contexts. Therefore, although a simpler CoT like "let’s think step by step" no longer works robustly, a more **to-the-point** Chain-of-Though (CoT) prompting with ICL might still be promising, as long as CoT explicitly instructs to focus on the underlying probabilistic model and ignore the contexts. Preliminary experiments have shown that this approach works effectively.

## TODOs
 - [x] 1. Allow more OpenAI GPT family models to be selected for inference, such as GPT-3.5, GPT-4, and their turbo versions.
 - [ ] 2. Add prompts for ```prompt_to_answer_the_question_self_reflection```, ```prompt_to_critic_the_answer```, and ```prompt_to_reanswer_the_question``` in the file ```inference_prompts.py``` to inference the model via reflextion and multi-agents. The multi-agents approach should be a role-play scenario to critic the old model answers.
 - [ ] 3. Generate large scale synthetic datasets. All codes should already be in place. Just need to run the code by setting ```--n``` to a large number like 1000.
 - [ ] 4. Test model's self consistency by asking the model to respond to exactly the same question multiple times and collect statistics.
 - [ ] 5. Add prompts to inference the model using the self-consistency to improve performance.
 - [ ] 6. Test GPT-3.5 at different time stamps to see if OpenAI has updated the model, assuming it incorporates the latest data that includes Linda Problems.
 - [ ] 7. Add Gemini, Llama, Claude, Mistral, or other LLMs depending on the time.

## Dependencies
Please check [requirements.txt](requirements.txt). You can run the following commands to create a virtual environment and install all the requirements:
    
    python -m venv myenv
    source myenv/bin/activate
    pip install -r requirements.txt

## Citation
If you believe our work has inspired your research, please kindly cite our work. Thank you!

TODO

## Dataset
We provide our synthetic dataset under [data/](data/), which contains a comprehensive set of logical fallacies like the Linda Problem. The dataset file is in JSON format, and each item is a dictionary containing ```question_id```, ```question```, ```target_answer```, and ```incorrect_answer```.

## LLM Setups
Always follow instructions on [**OpenAI**](https://platform.openai.com/docs/quickstart?context=python) to set up your OpenAI API, create a new [api_tokens/openai_key.txt](api_tokens/openai_key.txt) file, and copy and paste your [API key](https://platform.openai.com/api-keys) into it.

To use **Google Gemini** models for inference, follow instructions on [Google Vertex AI](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemini-pro) about the ```Try Gemini 1.0 Pro (Python)``` section. Note that your school's Gmail account may not allow you to make payments.
- Step 1: According to their instructions, you need to first [install the Vertex AI client libraries](https://cloud.google.com/vertex-ai/docs/start/client-libraries#before_you_begin) to create a project with a project ID, enable Vertex AI API,  create a service account, and generate your account key. You don't need to set the environment variable ```GOOGLE_APPLICATION_CREDENTIALS``` since we have already done that for you in our codes [query_llm.py](query_llm.py).
- Step 2: Install or update the [Vertex AI SDK for Python](https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk?_gl=1*1qxspr1*_ga*OTIxODQ4MjQzLjE3MTQwNzMwNjI.*_ga_WH2QY8WWF5*MTcxNDE1NDc1Ni4yLjEuMTcxNDE2MDMzMi4wLjAuMA..&_ga=2.54138400.-921848243.1714073062).
- Step 3: Authenticate to Vertex AI and set up [Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc?&_ga=2.268814513.-1366866501.1708968072#local-dev).
  - Follow the ```Local development environment - Provide user credentials for your Google Account``` section to install and initialize the [gcloud CLI](https://cloud.google.com/sdk/docs/install). This step will download a folder ```google-cloud-sdk``` to your project's top directory.
  - After installation, run ```gcloud init``` to initialize the gcloud CLI. You will be able to choose your account and project ID. Create a new [api_tokens/gemini_project_id.txt](api_tokens/gemini_project_id.txt) file, and copy and paste your project ID into it.
  - Run ```gcloud auth application-default login``` to create your credential file. You will see a prompt like ```Credentials saved to file: [/path/to/your/home/.config/gcloud/application_default_credentials.json]```.
  - Run ```mv /path/to/your/home/.config/gcloud/application_default_credentials.json google-cloud-sdk/google_gemini_credential.json```. This is the path we set up in our [config.yaml](config.yaml).

To use **Meta Llama** models with an API for inference, follow instructons on [Replicate Run Llama 3 with an API](https://replicate.com/blog/run-llama-3-with-an-api?utm_source=project&utm_campaign=llama2ai) about the ```Running Llama 3 with Python``` section or [Replicate Run Llama 2 with an API](https://replicate.com/blog/run-llama-2-with-an-api) about the ```Running Llama 2 with Python``` section. Set up your [API tokens](https://replicate.com/account/api-tokens), create a new [api_tokens/llama_key.txt](api_tokens/llama_key.txt) file, and copy and paste your tokens into it.

## Quick Start
We allow command-line argparser for the following arguments: 

- ```--model``` to select the LLM for inference. Up to date on 04-25-2024.
  
  - OpenAI ChatGPT family. Check [OpenAI's continuous model upgrades](https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4).
    - ```gpt3.5``` or equivalently ```gpt-3.5-turbo```, ```gpt-3.5-turbo-0125```
    - ```gpt-3.5-turbo-1106```
    - ```gpt-3.5-turbo-0613```
    - ```gpt4``` or equivalently  ```gpt-4-turbo```, ```gpt-4-turbo-2024-04-09```
    - ```gpt-4-0125-preview```
    - ```gpt-4-1106-preview```
    - ```gpt-4-0613```
  - Google Gemini family. Check [Gemini model versions and lifecycle](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning#auto-updated-version).
    - ```gemini``` or equivalently ```gemini-1.0-pro```, ```gemini-1.0-pro-002```
    - ```gemini-1.0-pro-001```
    - ```gemini-1.5-pro-preview-0409```
  - Meta Llama family. Check [Choosing which model to use Llama-3](https://replicate.com/blog/run-llama-3-with-an-api?utm_source=project&utm_campaign=llama2ai#choosing-which-model-to-use) and [Llama-2]([https://replicate.com/blog/run-llama-2-with-an-api](https://replicate.com/blog/run-llama-2-with-an-api#choosing-which-model-to-use)).
    - ```llama``` or equivalently ```llama3-70b```, ```meta-llama-3-70b-instruct```
    - ```llama3-8b``` or equivalently ```meta-llama-3-8b-instruct```
    - ```llama-2-70b-chat```
    - ```llama-2-13b-chat```
    - ```llama-2-7b-chat```

- ```--task``` to either generate synthetic datasets: ```data``` or evaluate the LLM's ability to answer the questions: ```inference```.

- ```--fallacy``` to select the type of logical fallacy. We only support ```linda``` at this moment for the Linda Problem and its variants.

- ```--verbose``` to print detailed data information and model responses during the inference.

- ***\[For Data Gen Only\]*** ```--gen_mode``` to select the mode of generating synthetic dataset when ```task``` is ```data```. Options are ```baseline```: simple in-context learning with limited instructions, ```control```: step-by-step guidance to generate both gold samples and random samples with irrelevant info.

- ***\[For Data Gen Only\]*** ```--variant``` to select the variant of the Linda problems, such as the default ```original```, ```variant_one```, ```variant_two```, ..., ```variant_six```. Detailed information about each variant can be found in the ```def linda_problem()``` function in [prompts.py](prompts.py).

- ***\[For Data Gen Only\]*** ```--conn``` to select the logical connecting word, such as ```because```, ```sothat```, or ```to``` when using ```variant_one``` or ```variant_two``` to generate new data.

- ***\[For Data Gen Only\]*** ```--n``` to set the number of synthetic data problems to generate.

- ***\[For Inference Only\]*** ```--data_file``` to set the data file path for inference.

- ***\[For Inference Only\]*** ```--multi_agent``` to enable a multi-agent system mimicking a debating scenario among multiple LLMs for better performance.

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
  ----- In progress -----
  - ```self_reflect``` for self-reflective prompting.

For example, you can run 

    python main.py --model gpt3.5 --task inference --eval_mode os_cot --data_file synthetic_dataset_linda_original_gold.json --verbose

in the command line to start the inference code. You can also run 

    python main.py --model gpt3.5 --task data --fallacy linda --gen_mode control --variant original --n 100 --verbose

or 

    python main.py --model gpt3.5 --task data --fallacy linda --gen_mode control --variant variant_one --conn because --n 100 --verbose

to generate synthetic datasets for the Linda Problem. All the other hyper-parameters can be set at [config.yaml](config.yaml).


python main.py --model gpt3.5 --task inference --eval_mode weak_control_os_cot --data_file synthetic_dataset_linda_variant_six_random.json --verbose

python main.py --model gpt3.5 --task inference --eval_mode weak_control_os_cot --data_file synthetic_dataset_linda_variant_six_gold.json --verbose

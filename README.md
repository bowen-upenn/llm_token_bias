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
TODO
    
## Dependencies
Please check [requirements.txt](requirements.txt). You can run the following commands to create a virtual environment and install all the requirements:
    
    python -m venv myenv
    source myenv/bin/activate
    pip install -r requirements.txt

## Citation
If you believe our work has inspired your research, please kindly cite our work. Thank you!

TODO

## Dataset
We provide our synthetic dataset under [data/](data/), which contains a comprehensive set of logical fallacies like the Linda Problem. The dataset file is in JSON format, and each item is a dictionary containing ```question_id```, ```question```, and ```target_answer```.

TODO

## Quick Start
We allow command-line argparser for the following arguments: 

- ```--model``` to select the LLM for inference: ```gpt3.5``` or equivalently ```gpt-3.5-turbo``` and ```gpt4``` or equivalently  ```gpt-4-turbo-preview```.

- ```--multi_agent``` to enable a multi-agent system mimicking a debating scenario among multiple LLMs for better performance.

- ```--task``` to either generate synthetic datasets: ```data``` or evaluate the LLM's ability to answer the questions: ```inference```.

- ```--fallacy``` to select the type of logical fallacy, such as ```linda```.

- ```--gen_mode``` to select the mode of generating synthetic dataset when ```task``` is ```data```. Options are ```baseline```: simple in-context learning with limited instructions, ```control```: step-by-step guidance to generate both gold samples and random samples with irrelevant info.

- ```--variant``` to select the variant of the Linda problems, such as the default ```original```, ```variant_one```, ```variant_two```, ..., ```variant_six```. Detailed information about each variant can be found in the ```def linda_problem()``` function in [prompts.py](prompts.py).

- ```--conn``` to select the logical connecting word, such as ```because```, ```sothat```, or ```to``` when using ```variant_one``` or ```variant_two``` to generate new data.

- ```--n``` to set the number of synthetic data problems to generate.

- ```--verbose``` to print detailed data information and model responses during the inference.

- ```--data_file``` to set the data file path for inference.

- ```--eval_mode``` to set the evaluation mode for the model to answer questions. Options are 

  - ```baseline``` for directly prompting, 
  - ```zs_cot``` for zero-shot chain-of-thought (CoT) prompting, 
  - ```os``` for one-shot in-context learning (ICL) prompting with the original Linda Problem (default), 
  - ```os_cot``` for one-shot ICL plus COT prompting , 
  - ```os_bob``` for one-shot ICL prompting but with a rephrased Bob Problem, 
  - ```os_bob_cot``` for one-shot ICL prompting plus COT but with a rephrased Bob Problem, 
  - ```os_incorrect``` for one-shot ICL but with an incorrect answer, 
  - ```os_incorrect_cot``` for one-shot ICL plus COT but with an incorrect answer,
  - ```fs``` for few-shot ICL prompting,
  - ```fs_cot``` for few-shot ICL plus COT prompting,
  - ```self_reflect``` for self-reflective prompting,
  - ```weak_control_zs_cot``` for weakly controlled zero-shot CoT prompting, leaking the hint that it is a Linda Problem but without detailed explanations,
  - ```weak_control_os_cot``` for weakly controlled one-shot CoT prompting, leaking the hint that it is a Linda Problem but without detailed explanations,
  - ```control_zs_cot``` for controlled zero-shot CoT prompting, leaking the hint that it is a Linda Problem with detailed and carefully-curated explanations,
  - ```control_os_cot``` for controlled one-shot CoT prompting, leaking the hint that it is a Linda Problem with detailed and carefully-curated explanations.

For example, you can run 

    python main.py --model gpt3.5 --task inference --eval_mode os_cot --data_file synthetic_dataset_linda_original_gold.json --verbose

in the command line to start the inference code. You can also run 

    python main.py --model gpt3.5 --task data --fallacy linda --gen_mode control --variant original --n 100 --verbose

or 

    python main.py --model gpt3.5 --task data --fallacy linda --gen_mode control --variant variant_one --conn because --n 100 --verbose

to generate synthetic datasets for the Linda Problem. All the other hyper-parameters can be set at [config.yaml](config.yaml).

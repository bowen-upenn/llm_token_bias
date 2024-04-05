## This is the official implementation of the paper "LLMs Are Not Yet Good Probabilistic Thinkers" in Pytorch.

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

- ```--gen_mode``` to select the mode of generating synthetic dataset when ```task``` is ```data```. Options are ```baseline```: simple in-context learning with limited instructions, ```gold```: step-by-step guidance, or ```random```: step-by-step guidance but the bio and hobby are irrelevent.

- ```--verbose``` to print detailed data information and model responses during the inference.

For example, you can run 

    python main.py --model gpt3.5 --task inference --multi_agent --verbose

in the command line to start the inference code. You can also run 

    python main.py --model gpt3.5 --task data --fallacy linda --gen_mode gold --verbose

to generate synthetic datasets for the Linda Problem. All the other hyper-parameters can be set at [config.yaml](config.yaml).

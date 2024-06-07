import torch
import random
from utils import *

class AllInferencePrompts:
    def __init__(self, args):
        self.args = args
        self.fallacy_type = args['datasets']['fallacy_type']
        self.linda_problem_variant = args['datasets']['linda_problem_variant']
        self.original_linda_problem = "Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. " \
                                      "As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations. " \
                                      "Which is more probable?\n" \
                                      "(a) Linda is a bank teller.\n" \
                                      "(b) Linda is a bank teller and is active in the feminist movement."

        self.bob_problem = "Bob is 29 years old, deeply passionate about environmental conservation, and volunteers his weekends at local park clean-ups. " \
                           "He studied environmental science in college, where he led a successful campaign to reduce the campus's carbon footprint. " \
                           "Bob is also an avid cyclist and promotes sustainable living practices whenever possible. " \
                           "Based on this information, which is more possible?\n" \
                           "(a) Bob works for a renewable energy company and is an active member of a local environmental advocacy group.\n" \
                           "(b) Bob works for a renewable energy company."

        self.syllogistic_fallacy_os = "Is this logically sound?\n" \
                                      "All roses are flowers.\n" \
                                      "Some flowers fade quickly.\n" \
                                      "Therefore some roses fade quickly."

        self.syllogistic_fallacy_os_rephrased = "Is this logically sound?\n" \
                                                "All carrots are vegetables.\n" \
                                                "Some vegetables are rich in fiber.\n" \
                                                "Therefore, some carrots are rich in fiber.\n",

        self.syllogistic_fallacy_fs = ["All carrots are vegetables.\nSome vegetables are rich in fiber.\nTherefore, some carrots are rich in fiber.\n",
                               "All roses are flowers. \nSome flowers fade quickly. \nTherefore some roses fade quickly.\n",
                               "All actors are performers.\nSome performers are skilled in improvisation.\nTherefore some actors are skilled in improvisation."]


    def load_all_data_entries(self):
        # exemplars are randomly selected from all synthetic datasets in diverse domains
        all_entries = load_all_data_entries_from_files(self.args['datasets']['data_dir'])
        self.all_entries = all_entries

    def select_random_few_shot_exemplars(self, num_exemplars):
        self.few_shot_exemplars = random.sample(self.all_entries, num_exemplars - 1)    # always add the original Linda problem


    ######## Prompts to evaluate the model's ability in answering the problems in the synthetic dataset ########
    def prompt_to_answer_the_question_directly(self, question):
        if self.fallacy_type == 'linda':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a), (b), etc."},
                {"role": "user", "content": question}
            ]
        elif self.fallacy_type == 'sets':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly saying 'Yes' or 'No'."},
                {"role": "user", "content": question}
            ]
        else:
            raise ValueError("Invalid fallacy type: " + self.fallacy_type)
        return message

    def prompt_to_answer_the_question_zero_shot_cot(self, question):
        if self.fallacy_type == 'linda':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a), (b), etc."},
                {"role": "user", "content": question + "\nLet’s think step by step."}
            ]
        elif self.fallacy_type == 'sets':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly saying 'Yes' or 'No'."},
                {"role": "user", "content": question + "\nLet’s think step by step."}
            ]
        else:
            raise ValueError("Invalid fallacy type: " + self.fallacy_type)
        return message

    def prompt_to_answer_the_question_one_shot(self, question):
        if self.fallacy_type == 'linda':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a), (b), etc. Here is an example."},
                {"role": "user", "content": self.original_linda_problem},
                {"role": "assistant", "content": "The correct answer is (a) Linda is a bank teller."},
                {"role": "user", "content": "Here is another question:\n" + question}
            ]
        elif self.fallacy_type == 'sets':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly saying 'Yes' or 'No'. Here is an example."},
                {"role": "user", "content": self.syllogistic_fallacy_os},
                {"role": "assistant", "content": "The correct answer is No."},
                {"role": "user", "content": "Here is another question:\n" + question}
            ]
        else:
            raise ValueError("Invalid fallacy type: " + self.fallacy_type)
        return message

    def prompt_to_answer_the_question_one_shot_cot(self, question):
        if self.fallacy_type == 'linda':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a), (b), etc. Here is an example."},
                {"role": "user", "content": self.original_linda_problem},
                {"role": "assistant", "content": "The correct answer is (a) Linda is a bank teller."},
                {"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."}
            ]
        elif self.fallacy_type == 'sets':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly saying 'Yes' or 'No'. Here is an example."},
                {"role": "user", "content": self.syllogistic_fallacy_os},
                {"role": "assistant", "content": "The correct answer is No."},
                {"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."}
            ]
        else:
            raise ValueError("Invalid fallacy type: " + self.fallacy_type)
        return message

    def prompt_to_answer_the_question_one_shot_bob(self, question):
        if self.fallacy_type == 'linda':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a), (b), etc. Here is an example."},
                {"role": "user", "content": self.bob_problem},
                {"role": "assistant", "content": "The correct answer is (b) Bob works for a renewable energy company."},
                {"role": "user", "content": "Here is another question:\n" + question}
            ]
        else:
            raise ValueError("Invalid prompt for the fallacy type: " + self.fallacy_type)
        return message

    def prompt_to_answer_the_question_one_shot_bob_cot(self, question):
        if self.fallacy_type == 'linda':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a), (b), etc. Here is an example."},
                {"role": "user", "content": self.bob_problem},
                {"role": "assistant", "content": "The correct answer is (b) Bob works for a renewable energy company."},
                {"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."}
            ]
        else:
            raise ValueError("Invalid prompt for the fallacy type: " + self.fallacy_type)
        return message

    def prompt_to_answer_the_question_one_shot_incorrect_answer(self, question):
        if self.fallacy_type == 'linda':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a), (b), etc. Here is an example."},
                {"role": "user", "content": self.bob_problem},
                {"role": "assistant", "content": "The correct answer is (b) Linda is a bank teller and is active in the feminist movement."},
                {"role": "user", "content": "Here is another question:\n" + question}
            ]
        elif self.fallacy_type == 'sets':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly saying 'Yes' or 'No'. Here is an example."},
                {"role": "user", "content": self.syllogistic_fallacy_os_rephrased},
                {"role": "assistant", "content": "The correct answer is Yes."},
                {"role": "user", "content": "Here is another question:\n" + question}
            ]
        else:
            raise ValueError("Invalid fallacy type: " + self.fallacy_type)
        return message

    def prompt_to_answer_the_question_one_shot_incorrect_answer_cot(self, question):
        if self.fallacy_type == 'linda':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a), (b), etc. Here is an example."},
                {"role": "user", "content": self.bob_problem},
                {"role": "assistant", "content": "The correct answer is (b) Linda is a bank teller and is active in the feminist movement."},
                {"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."}
            ]
        elif self.fallacy_type == 'sets':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly saying 'Yes' or 'No'. Here is an example."},
                {"role": "user", "content": self.syllogistic_fallacy_os_rephrased},
                {"role": "assistant", "content": "The correct answer is Yes."},
                {"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."}
            ]
        else:
            raise ValueError("Invalid fallacy type: " + self.fallacy_type)
        return message

    def prompt_to_answer_the_question_few_shots(self, question):
        if self.fallacy_type == 'linda':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a), (b), etc. Here are some examples."},
                {"role": "user", "content": self.original_linda_problem},
                {"role": "assistant", "content": "The correct answer is (a) Linda is a bank teller."},
            ]

            for exemplar in self.few_shot_exemplars:
                message.append({"role": "user", "content": exemplar['question']})
                message.append({"role": "assistant", "content": "The correct answer is " + exemplar['target_answer']})

            message.append({"role": "user", "content": "Here is another question:\n" + question})

        elif self.fallacy_type == 'sets':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly saying 'Yes' or 'No'. Here are some examples."}
            ]

            for exemplar in self.syllogistic_fallacy_fs:
                message.append({"role": "user", "content": "Is this logically sound?\n" + exemplar})
                message.append({"role": "assistant", "content": "The correct answer is No."})

            message.append({"role": "user", "content": "Here is another question:\n" + question})
        else:
            raise ValueError("Invalid fallacy type: " + self.fallacy_type)
        return message

    def prompt_to_answer_the_question_few_shots_cot(self, question):
        if self.fallacy_type == 'linda':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a), (b), etc. Here are some examples."},
                {"role": "user", "content": self.original_linda_problem},
                {"role": "assistant", "content": "The correct answer is (a) Linda is a bank teller."},
            ]

            for exemplar in self.few_shot_exemplars:
                message.append({"role": "user", "content": exemplar['question']})
                message.append({"role": "assistant", "content": "The correct answer is " + exemplar['target_answer']})

            message.append({"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."})

        elif self.fallacy_type == 'sets':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly saying 'Yes' or 'No'. Here are some examples."}
            ]

            for exemplar in self.syllogistic_fallacy_fs:
                message.append({"role": "user", "content": "Is this logically sound?\n" + exemplar})
                message.append({"role": "assistant", "content": "The correct answer is No."})

            message.append({"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."})
        else:
            raise ValueError("Invalid fallacy type: " + self.fallacy_type)
        return message

    def prompt_to_answer_the_question_self_reflection(self, question):
        return message

    def prompt_to_answer_the_question_prob_zero_shot_cot(self, question):
        # in weakly controlled experiments, we only tell the model that it is a Linda Problem, without explaining what Linda Problem is and the reasoning behind it
        if self.fallacy_type == 'linda':
            message = [
                {"role": "system", "content": "You are a rational probabilistic thinker. Please answer the following question by explicitly selecting either option (a), (b), etc. "},
                {"role": "user", "content": question + "\nLet’s think step by step."}
            ]
        elif self.fallacy_type == 'sets':
            message = [
                {"role": "system", "content": "You are a rational probabilistic thinker. Please answer the following question by explicitly saying 'Yes' or 'No'. "},
                {"role": "user", "content": question + "\nLet’s think step by step."}
            ]
        else:
            raise ValueError("Invalid fallacy type: " + self.fallacy_type)
        return message

    def prompt_to_answer_the_question_prob_one_shot_cot(self, question):
        # in weakly controlled experiments, we only tell the model that it is a Linda Problem, without explaining what Linda Problem is and the reasoning behind it
        if self.fallacy_type == 'linda':
            message = [
                {"role": "system", "content": "You are a rational probabilistic thinker. Please answer the following question by explicitly selecting either option (a), (b), etc. Here is an example."},
                {"role": "user", "content": self.original_linda_problem},
                {"role": "assistant", "content": "The correct answer is (a) Linda is a bank teller."},
                {"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."}
            ]
        elif self.fallacy_type == 'sets':
            message = [
                {"role": "system", "content": "You are a rational probabilistic thinker. Please answer the following question by explicitly saying 'Yes' or 'No'. Here is an example."},
                {"role": "user", "content": self.syllogistic_fallacy_os},
                {"role": "assistant", "content": "The correct answer is No."},
                {"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."}
            ]
        else:
            raise ValueError("Invalid fallacy type: " + self.fallacy_type)
        return message

    def prompt_to_answer_the_question_weak_control_zero_shot_cot(self, question):
        # in weakly controlled experiments, we only tell the model that it is a Linda Problem, without explaining what Linda Problem is and the reasoning behind it
        if self.fallacy_type == 'linda':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a), (b), etc. "
                                              "Please aware that this is a Linda Problem designed to explore the concept of the conjunction fallacy."},
                {"role": "user", "content": question + "\nLet’s think step by step."}
            ]
        elif self.fallacy_type == 'sets':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly saying 'Yes' or 'No'. "
                                              "Please aware that this is a Linda Problem designed to explore the concept of the conjunction fallacy."},
                {"role": "user", "content": question + "\nLet’s think step by step."}
            ]
        else:
            raise ValueError("Invalid fallacy type: " + self.fallacy_type)
        return message

    def prompt_to_answer_the_question_weak_control_one_shot_cot(self, question):
        # in weakly controlled experiments, we only tell the model that it is a Linda Problem, without explaining what Linda Problem is and the reasoning behind it
        if self.fallacy_type == 'linda':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a), (b), etc. "
                                              "Please aware that this is a Linda Problem designed to explore the concept of the conjunction fallacy. Here is an example."},
                {"role": "user", "content": self.original_linda_problem},
                {"role": "assistant", "content": "The correct answer is (a) Linda is a bank teller."},
                {"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."}
            ]
        elif self.fallacy_type == 'sets':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly saying 'Yes' or 'No'. "
                                              "Please aware that this is a Linda Problem designed to explore the concept of the conjunction fallacy. Here is an example."},
                {"role": "user", "content": self.syllogistic_fallacy_os},
                {"role": "assistant", "content": "The correct answer is No."},
                {"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."}
            ]
        else:
            raise ValueError("Invalid fallacy type: " + self.fallacy_type)
        return message

    def prompt_to_answer_the_question_control_zero_shot_cot(self, question):
        if self.fallacy_type == 'linda':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a), (b), etc.\n"
                                              "Please aware that this is a Linda Problem designed to explore the concept of the conjunction fallacy. "
                                              "The conjunction fallacy occurs when individuals incorrectly judge the conjunction of two events as more probable than one of the events alone. "
                                              "For instance, many might believe that Linda, who is described as a bright, single woman deeply concerned with discrimination and social justice, "
                                              "is more likely to be both a bank teller and active in the feminist movement than just a bank teller. "
                                              "This judgment violates the basic probability rule: the probability of a conjunction, "
                                              "P(A and B), is always less than or equal to the probabilities of its constituents, P(A) or P(B). "
                                              "This error often stems from the representativeness heuristic, "
                                              "where people estimate the likelihood of an event by how closely it matches their mental prototype.\n"
                                              "To correctly solve problems like this, you must adopt probabilistic thinking: "
                                              "abstract the problem from its narrative context and focus solely on the probabilistic models. Ignore all extraneous background information and "
                                              "consistently choose the option involving a single event as it statistically holds a higher likelihood than the conjunction of multiple events."},
                {"role": "user", "content": "Here is the question:\n" + question + "\nLet’s think step by step."}
            ]
        elif self.fallacy_type == 'sets':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly saying 'Yes' or 'No'.\n"
                                              "Please aware that this is a Syllogistic Fallacy Problem. This type of reasoning is known as a syllogism. "
                                              "For a syllogism to be valid, the conclusion must necessarily follow from the premises. "
                                              "Pay close attention to quantifiers such as 'All', 'Some', 'No', or similar terms. These terms help define the distribution of properties or elements within the given groups or categories in the premises. "
                                              "Next, assess whether the attribute ascribed in the conclusion necessarily follows from the attributes described in the premises. "
                                              "Consider if the subset described in the second premise encompasses or overlaps with the elements in the first premise that are carried into the conclusion. "
                                              "A common pitfall in syllogistic reasoning is the erroneous assumption that a characteristic of a subset of a group (from the premises) applies to another "
                                              "subset of the same or different group (in the conclusion), without explicit justification."
                                              "Here is the question:\n" + question + "\nLet’s think step by step."}
            ]
        else:
            raise ValueError("Invalid fallacy type: " + self.fallacy_type)
        return message

    def prompt_to_answer_the_question_control_one_shot_cot(self, question):
        if self.fallacy_type == 'linda':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a), (b), etc.\n"
                                              "Please aware that this is a Linda Problem designed to explore the concept of the conjunction fallacy. "
                                              "The conjunction fallacy occurs when individuals incorrectly judge the conjunction of two events as more probable than one of the events alone. "
                                              "For instance, many might believe that Linda, who is described as a bright, single woman deeply concerned with discrimination and social justice, "
                                              "is more likely to be both a bank teller and active in the feminist movement than just a bank teller. "
                                              "This judgment violates the basic probability rule: the probability of a conjunction, "
                                              "P(A and B), is always less than or equal to the probabilities of its constituents, P(A) or P(B). "
                                              "This error often stems from the representativeness heuristic, "
                                              "where people estimate the likelihood of an event by how closely it matches their mental prototype.\n"
                                              "To correctly solve problems like this, you must adopt probabilistic thinking: "
                                              "abstract the problem from its narrative context and focus solely on the probabilistic models. Ignore all extraneous background information and "
                                              "consistently choose the option involving a single event as it statistically holds a higher likelihood than the conjunction of multiple events.\n"
                                              "Here is an example."},
                {"role": "user", "content": self.original_linda_problem},
                {"role": "assistant", "content": "The correct answer is (a) Linda is a bank teller, "
                                                 "because the probability of a single event occurring is always higher than the probability of it occurring in conjunction with another specific event, "
                                                 "making option (a) statistically more probable than option (b)."},
                {"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."}
            ]
        elif self.fallacy_type == 'sets':
            message = [
                {"role": "system", "content": "Your task is to answer the following question by explicitly saying 'Yes' or 'No'.\n"
                                              "Please aware that this is a Syllogistic Fallacy Problem. This type of reasoning is known as a syllogism. "
                                              "For a syllogism to be valid, the conclusion must necessarily follow from the premises. "
                                              "Pay close attention to quantifiers such as 'All', 'Some', 'No', or similar terms. These terms help define the distribution of properties or elements within the given groups or categories in the premises. "
                                              "Next, assess whether the attribute ascribed in the conclusion necessarily follows from the attributes described in the premises. "
                                              "Consider if the subset described in the second premise encompasses or overlaps with the elements in the first premise that are carried into the conclusion. "
                                              "A common pitfall in syllogistic reasoning is the erroneous assumption that a characteristic of a subset of a group (from the premises) applies to another "
                                              "subset of the same or different group (in the conclusion), without explicit justification."
                                              "Here is an example."},
                {"role": "user", "content": self.syllogistic_fallacy_os},
                {"role": "assistant", "content": "The correct answer is No. The conclusion does not necessarily follow from the premises because the fact that some flowers fade quickly doesn't specifically imply that any of those flowers are roses. "
                                                 "The flowers that fade quickly could be a different type of flower."},
                {"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."}
            ]
        else:
            raise ValueError("Invalid fallacy type: " + self.fallacy_type)
        return message


    def prompt_to_critic_the_answer(self, question, model_answer):
        message = [
            {"role": "system", "content": ""},  # add instruction and question here
            {"role": "user", "content": ""}  # add model response here
        ]  # Note: ask the model to attach special tokens in the response, such as [Yes] or [No]
        return message


    def prompt_to_reanswer_the_question(self, question, init_model_answer, critic):
        message = [
            {"role": "system", "content": ""},  # add initial instruction here
            {"role": "user", "content": ""},  # add initial question here
            {"role": "system", "content": ""},  # add initial model response here
            {"role": "user", "content": ""},  # add critic here
            {"role": "system", "content": ""}  # add reattempt instruction here
        ]
        return message


    def prompt_to_grade_the_answer(self, target_answer, model_answer, grader_id=0):
        if grader_id == 0:
            messages = [
                {"role": "system", "content": "Your task is to evaluate whether the model's answer corresponds to the correct option provided. "
                                              "The model's response does not need to precisely match the target answer; "
                                              "but it should accurately align with the option (e.g., (a), (b), etc.) that is correct. "
                                              "You MUST begin your evaluation with either [Correct] or [Incorrect]. No need to provide any reasoning."},
                {"role": "user", "content": "The ground-truth answer is: " + target_answer + "\nHere is the model's predicted answer:\n" + model_answer}
            ]
        elif grader_id == 1:
            messages = [
                {"role": "system", "content": "To assess the model's response, determine if it correctly corresponds to the given option. "
                                              "While an exact match with the target answer isn't necessary, it must align with the correct option "
                                              "(e.g., (a), (b), etc.). Your evaluation must come with either [Correct] or [Incorrect]. No need to explain."},
                {"role": "user", "content": "The ground-truth answer is: " + target_answer + "\nHere is the model's predicted answer:\n" + model_answer}
            ]
        else:
            messages = [
                {"role": "system", "content": "Begin your evaluation with either [Correct] or [Incorrect]. "
                                              "Please assess whether the model's answer corresponds to the given correct option. While the model's response need not "
                                              "exactly mirror the target answer, it should match accurately with the option (e.g., (a), (b), etc.)."},
                {"role": "user", "content": "The ground-truth answer is: " + target_answer + "\nHere is the model's predicted answer:\n" + model_answer}
            ]
        return messages


    def prompt_to_extract_the_answer(self, model_answer):
        message = [
            {"role": "system", "content": "Given the detailed response below that might include reasoning for each option in a multiple choice question, "
                                          "please identify and return the final answer chosen."},
            {"role": "user", "content": "Detailed Response:\n" + model_answer + "\nWhat is the final answer to the question? The final option (a), (b), or etc only."}
        ]
        return message

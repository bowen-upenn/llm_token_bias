import torch
import random
from utils import *

class AllInferencePrompts:
    def __init__(self, args):
        self.args = args
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


    def load_all_data_entries(self):
        # exemplars are randomly selected from all synthetic datasets in diverse domains
        all_entries = load_all_data_entries_from_files(self.args['datasets']['data_dir'])
        self.all_entries = all_entries

    def select_random_few_shot_exemplars(self, num_exemplars):
        self.few_shot_exemplars = random.sample(self.all_entries, num_exemplars - 1)    # always add the original Linda problem


    ######## Prompts to evaluate the model's ability in answering the problems in the synthetic dataset ########
    def prompt_to_answer_the_question_directly(self, question):
        message = [
            {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a) or (b)."},
            {"role": "user", "content": question}
        ]
        return message

    def prompt_to_answer_the_question_zero_shot_cot(self, question):
        message = [
            {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a) or (b)."},
            {"role": "user", "content": question + "\nLet’s think step by step."}
        ]
        return message

    def prompt_to_answer_the_question_one_shot(self, question):
        message = [
            {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a) or (b). Here is an example."},
            {"role": "user", "content": self.original_linda_problem()},
            {"role": "assistant", "content": "The correct answer is (a) Linda is a bank teller."},
            {"role": "user", "content": "Here is another question:\n" + question}
        ]
        return message

    def prompt_to_answer_the_question_one_shot_cot(self, question):
        message = [
            {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a) or (b). Here is an example."},
            {"role": "user", "content": self.original_linda_problem()},
            {"role": "assistant", "content": "The correct answer is (a) Linda is a bank teller."},
            {"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."}
        ]
        return message

    def prompt_to_answer_the_question_one_shot_bob(self, question):
        message = [
            {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a) or (b). Here is an example."},
            {"role": "user", "content": self.bob_problem()},
            {"role": "assistant", "content": "The correct answer is (b) Bob works for a renewable energy company."},
            {"role": "user", "content": "Here is another question:\n" + question}
        ]
        return message

    def prompt_to_answer_the_question_one_shot_bob_cot(self, question):
        message = [
            {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a) or (b). Here is an example."},
            {"role": "user", "content": self.bob_problem()},
            {"role": "assistant", "content": "The correct answer is (b) Bob works for a renewable energy company."},
            {"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."}
        ]
        return message

    def prompt_to_answer_the_question_one_shot_incorrect_answer(self, question):
        message = [
            {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a) or (b). Here is an example."},
            {"role": "user", "content": self.original_linda_problem()},
            {"role": "assistant", "content": "The correct answer is (b) Linda is a bank teller and is active in the feminist movement."},
            {"role": "user", "content": "Here is another question:\n" + question}
        ]
        return message

    def prompt_to_answer_the_question_one_shot_incorrect_answer_cot(self, question):
        message = [
            {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a) or (b). Here is an example."},
            {"role": "user", "content": self.original_linda_problem()},
            {"role": "assistant", "content": "The correct answer is (b) Linda is a bank teller and is active in the feminist movement."},
            {"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."}
        ]
        return message

    def prompt_to_answer_the_question_few_shot(self, question):
        message = [
            {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a) or (b). Here are some examples."},
            {"role": "user", "content": self.original_linda_problem()},
            {"role": "assistant", "content": "The correct answer is (a) Linda is a bank teller."},
        ]

        for exemplar in self.few_shot_exemplars:
            message.append({"role": "user", "content": exemplar['question']})
            message.append({"role": "assistant", "content": "The correct answer is " + exemplar['target_answer']})

        message.append({"role": "user", "content": "Here is another question:\n" + question})
        return message

    def prompt_to_answer_the_question_few_shot_cot(self, question):
        message = [
            {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a) or (b). Here are some examples."},
            {"role": "user", "content": self.original_linda_problem()},
            {"role": "assistant", "content": "The correct answer is (a) Linda is a bank teller."},
        ]

        for exemplar in self.few_shot_exemplars:
            message.append({"role": "user", "content": exemplar['question']})
            message.append({"role": "assistant", "content": "The correct answer is " + exemplar['target_answer']})

        message.append({"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."})

        return message

    def prompt_to_answer_the_question_self_reflection(self, question):
        return message

    def prompt_to_answer_the_question_weak_control_zero_shot_cot(self, question):
        # in weakly controlled experiments, we only tell the model that it is a Linda Problem, without explaining what Linda Problem is and the reasoning behind it
        message = [
            {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a) or (b). "
                                          "Please aware that this is a Linda Problem designed to explore the concept of the conjunction fallacy."},
            {"role": "user", "content": question + "\nLet’s think step by step."}
        ]
        return message

    def prompt_to_answer_the_question_weak_control_one_shot_cot(self, question):
        # in weakly controlled experiments, we only tell the model that it is a Linda Problem, without explaining what Linda Problem is and the reasoning behind it
        message = [
            {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a) or (b). "
                                          "Please aware that this is a Linda Problem designed to explore the concept of the conjunction fallacy. Here is an example."},
            {"role": "user", "content": self.original_linda_problem()},
            {"role": "assistant", "content": "The correct answer is (a) Linda is a bank teller."},
            {"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."}
        ]
        return message

    def prompt_to_answer_the_question_control_zero_shot_cot(self, question):
        message = [
            {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a) or (b).\n"
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
            {"role": "user", "content": question + "\nLet’s think step by step."}
        ]
        return message

    def prompt_to_answer_the_question_control_one_shot_cot(self, question):
        message = [
            {"role": "system", "content": "Your task is to answer the following question by explicitly selecting either option (a) or (b).\n"
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
            {"role": "user", "content": self.original_linda_problem()},
            {"role": "assistant", "content": "The correct answer is (a) Linda is a bank teller, "
                                             "because the probability of a single event occurring is always higher than the probability of it occurring in conjunction with another specific event, "
                                             "making option (a) statistically more probable than option (b)."},
            {"role": "user", "content": "Here is another question:\n" + question + "\nLet’s think step by step."}
        ]
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


    def prompt_to_grade_the_answer(self, question, target_answer, model_answer, grader_id=0):
        if grader_id == 0:
            messages = [
                {"role": "system", "content": ""},  # add instruction here
                {"role": "user", "content": ""},  # add model response here
            ]
        elif grader_id == 1:
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": ""},
            ]
        else:
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": ""},
            ]
        return messages


    def prompt_to_generate_synthetic_data(self, question):
        message = [
            {"role": "system", "content": ""},  # add instruction here
            {"role": "user", "content": ""}  # add in-context learning example here
        ]
        return message
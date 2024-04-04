import torch
import random
from utils import load_occupations

class AllPrompts:
    def __init__(self, args):
        self.args = args
        self.original_linda_problem = self.linda_problem()
        self.all_occupations = load_occupations(args['datasets']['occupations_filename'])
        self.random_occupation = 'bank teller'
        self.random_gender = 'female'
        self.random_age = 31
        self.random_race = 'asian'

    def select_a_random_occupation(self):
        self.random_occupation = random.choice(self.all_occupations)

    def select_a_random_gender(self):
        self.random_gender = random.choice(['female', 'male', 'female', 'male', 'nonbinary', 'transgender', 'queer'])

    def select_a_random_age(self):
        self.random_age = random.randint(25, 50)

    def select_a_random_race(self):
        self.random_race = random.choice(['white', 'black', 'brown', 'asian', 'latino', 'native american', 'pacific islander'])


    ######## The original Linda problem ########
    def linda_problem(self):
        return "Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. " \
               "As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations. " \
               "Which is more probable?\n" \
               "(a) Linda is a bank teller.\n" \
               "(b) Linda is a bank teller and is active in the feminist movement."


    ######## Prompts to create other Linda problems into a synthetic dataset ########
    def prompt_to_create_linda_problems_baseline(self):
        message = [
            {"role": "system",
             "content": "Your task is to create another Linda problem following the example below.\n" + self.original_linda_problem + "\n"
                        "Write another example here:"}
        ]
        return message


    def prompt_to_write_a_bio(self):
        message = [
            {"role": "system",
             "content": "Your task is to write a short bio for a random person within 100 words. "
                        "You shall pick a random name, use gender " + self.random_gender + ", race " + self.random_race + ", and an age " + str(self.random_age) + ". "
                        "The bio should describe the college majors, some personal characters, and interests. Keep the bio short.\n"
                        "For example, 'Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. "
                        "As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations.'\n"
                        "Write another example here:"}
        ]
        return message


    def prompt_to_find_a_hobby(self, previous_response):
        message = [
            {"role": "system",
             "content": "Your task is to write a short bio for a random person within 100 words. "
                        "You shall pick a random name, use gender " + self.random_gender + ", race " + self.random_race + ", and an age " + str(self.random_age) + ". "
                        "The bio should describe the college majors, some personal characters, and interests. Keep the bio short.\n"
                        "For example, 'Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. "
                        "As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations.'\n"
                        "Write another example here:"},
            {"role": "assistant", "content": previous_response},
            {"role": "user",
             "content": "Your next step is to find a hobby or activity that the person mentioned before will be interested in based on your experience. "
                        "The hobby or activity must be relevant to the bio descriptions. "
                        "In the example above, we can say that 'Linda is active in the feminist movement.' because her bio says she was concerned with discrimination and social justice. "
                        "Please keep your answer in one sentence and begin with that person's name, "
                        "but refrain from using any words used in the bio."}
        ]
        return message


    def prompt_to_find_a_irrelevant_hobby(self):   # control study
        message = [
            {"role": "system",
             "content": "Your task is to find a random hobby or activity, and keep your answer short in one sentence. For example, you can say 'cook Asian foods.'"}
        ]
        return message


    def prompt_to_create_linda_problems(self, previous_response_bio, previous_response_hobby):
        message = [
            {"role": "system",
             "content": "Your task is to write a short bio for a random person within 100 words. "
                        "You shall pick a random name, use gender " + self.random_gender + ", race " + self.random_race + ", and an age " + str(self.random_age) + ". "
                        "The bio should describe the college majors, some personal characters, and interests. Keep the bio short.\n"
                        "For example, 'Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. "
                        "As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations.'\n"
                        "Write another example here:"},
            {"role": "assistant", "content": previous_response_bio},
            {"role": "user",
             "content": "Your next step is to find a hobby or activity that the person will be interested in based on your experience. "
                        "The hobby or activity must be relevant to the bio descriptions. "
                        "In the example above, we can say that 'Linda is active in the feminist movement.' "
                        "Please keep your answer in one sentence and begin with that person's name, like 'Linda is', "
                        "but refrain from using any words used in the bio."},
            {"role": "assistant", "content": previous_response_hobby},
            {"role": "user",
             "content": "Here is an example of a complete Linda problem: " + self.original_linda_problem + "\n"
                        "Your final step is to summarize the bio and hobby mentioned in your previous responses in the format of a Linda problem. "
                        "The problem should have two options (a) and (b), one of which should be a subset of the other, and you are allowed to switch the order. "
                        "The problem statement should exactly match the bio. "
                        "Use the employment occupation '" + self.random_occupation + "' for both options, "
                        "except that the hobby " + previous_response_hobby + " should also be included in the longer option only. "
                        "Do not make any changes to the bio or the hobby."}
        ]
        return message


    def prompt_to_create_linda_problems_irrelevant(self, previous_response_bio, previous_response_hobby):
        message = [
            {"role": "system",
             "content": "Your task is to write a short bio for a random person within 100 words. "
                        "You shall pick a random name, use gender " + self.random_gender + ", race " + self.random_race + ", and an age " + str(self.random_age) + ". "
                        "The bio should describe the college majors, some personal characters, and interests. Keep the bio short.\n"
                        "For example, 'Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. "
                        "As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations.'\n"
                        "Write another example here:"},
            {"role": "assistant", "content": previous_response_bio},
            {"role": "user",
             "content": "Your task is to find a random hobby or activity, and keep your answer short in one sentence. For example, you can say 'cook Asian foods.'"},
            {"role": "assistant", "content": previous_response_hobby},
            {"role": "user",
             "content": "Here is an example of a complete Linda problem: " + self.original_linda_problem + "\n"
                        "Your final step is to summarize the bio and hobby mentioned in your previous responses in the format of a Linda problem. "
                        "The problem should have two options (a) and (b), one of which should be a subset of the other, and you are allowed to switch the order. "
                        "The problem statement should exactly match the bio. "
                        "Use the employment occupation '" + self.random_occupation + "' for both options, "
                        "except that the hobby " + previous_response_hobby + " should also be included in the longer option only. "
                        "Do not make any changes to the bio or the hobby."}
        ]
        return message


    ######## Prompts to evaluate the model's ability in answering the problems in the synthetic dataset ########
    def prompt_to_answer_the_question(self, question):
        message = [
            {"role": "system", "content": ""},  # add instruction here
            {"role": "user", "content": ""}  # add question here
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
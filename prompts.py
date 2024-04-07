import torch
import random
from utils import *

class AllPrompts:
    def __init__(self, args):
        self.args = args
        self.linda_problem_variant = args['datasets']['linda_problem_variant']
        self.original_linda_problem = self.linda_problem()

        if self.linda_problem_variant == 'original':
            self.all_occupations = load_occupations(args['datasets']['occupations_filename'])
            self.random_occupation = 'bank teller'
            self.random_gender = 'female'
            self.random_age = 31
            self.random_race = 'asian'
        elif self.linda_problem_variant == 'variant_one':
            self.all_roc_stories = load_roc_stories(args['datasets']['roc_stories_filename'])
            self.connector = 'because'
        elif self.linda_problem_variant == 'variant_two':
            self.all_news = load_cnn_dailymails(args['datasets']['cnn_dailymails_filename'])
            self.connector = 'because'

    def select_a_random_occupation(self):
        self.random_occupation = random.choice(self.all_occupations)

    def select_a_random_gender(self):
        self.random_gender = random.choice(['female', 'male', 'female', 'male', 'nonbinary', 'transgender', 'queer'])

    def select_a_random_age(self):
        self.random_age = random.randint(25, 50)

    def select_a_random_race(self):
        self.random_race = random.choice(['white', 'black', 'african american', 'brown', 'asian', 'latino', 'native american', 'pacific islander'])

    def select_a_random_roc_story(self):
        self.random_roc_story = random.choice(self.all_roc_stories)

    def select_a_random_news(self):
        self.random_news = random.choice(self.all_news).split('.')
        while len(self.random_news) <= 1:
            self.random_news = random.choice(self.all_news).split('.')
        self.random_news_last_sentence = self.random_news[-2]
        self.random_news_before_last_sentence = '. '.join(self.random_news[:-2])
        print('news', self.random_news_before_last_sentence, 'last', self.random_news_last_sentence)


    ######## The original Linda problem ########
    def linda_problem(self):
        if self.linda_problem_variant == 'original':
            return "Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. " \
                   "As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations. " \
                   "Which is more probable?\n" \
                   "(a) Linda is a bank teller.\n" \
                   "(b) Linda is a bank teller and is active in the feminist movement."

        elif self.linda_problem_variant == 'variant_one':
            return "John P. is a meek man, 42 years old, married with two children. His neighbors describe him as mild-mannered but somewhat secretive. " \
                   "He owns an import-export company based in New York City, and he travels frequently to Europe and the Far East. " \
                   "Mr. P. was convicted once for smuggling precious stones and metals (including uranium) and received a suspended sentence of 6 months in jail and a large fine. " \
                   "Mr. P. is currently under police investigation. " \
                   "Which one is more likely?\n" \
                   "(a) Mr. P. killed one of his employees.\n" \
                   "(b) Mr. P. killed one of his employees because the employee discovered Mr. P. was involved in illegal smuggling activities."

        elif self.linda_problem_variant == 'variant_two':
            return "John P. is a meek man, 42 years old, married with two children. His neighbors describe him as mild-mannered but somewhat secretive. " \
                   "He owns an import-export company based in New York City, and he travels frequently to Europe and the Far East. " \
                   "Mr. P. was convicted once for smuggling precious stones and metals (including uranium) and received a suspended sentence of 6 months in jail and a large fine. " \
                   "Mr. P. is currently under police investigation. " \
                   "Which one is more likely?\n" \
                   "(a) Mr. P. killed one of his employees.\n" \
                   "(b) Mr. P. killed one of his employees to prevent him from talking to the police."


    ######## Prompts to create other Linda problems, original version ########
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

    def prompt_to_create_linda_problems_original(self, previous_response_bio, previous_response_hobby):
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
                        "You shall include the question 'Which is more probable?' followed by two options (a) and (b), one of which should be a subset of the other, and you are allowed to switch the order. "
                        "The problem statement should exactly match the bio. "
                        "Use the employment occupation '" + self.random_occupation + "' for both options, "
                        "except that the hobby " + previous_response_hobby + " should also be included in the longer option only. "
                        "Do not make any changes to the bio or the hobby.\n Here is the new problem:"}
        ]
        return message

    def prompt_to_create_linda_problems_original_irrelevant(self, previous_response_bio, previous_response_hobby):
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
                        "You shall include the question 'Which is more probable?' followed by two options (a) and (b), one of which should be a subset of the other, and you are allowed to switch the order. "
                        "The problem statement should exactly match the bio. "
                        "Use the employment occupation '" + self.random_occupation + "' for both options, "
                        "except that the hobby " + previous_response_hobby + " should also be included in the longer option only. "
                        "Do not make any changes to the bio or the hobby.\n Here is the new problem:"}
        ]
        return message


    ######## Prompts to create other Linda problems, variant one ########
    def prompt_to_extend_the_story(self):
        message = [
            {"role": "system",
             "content": "What is a likely thing that the protagonist in the following short story would do? One sentence only. "
                        "Do not add any reasons or purposes and avoid words like 'because' or 'to'. \n" + self.random_roc_story}
        ]
        return message

    def prompt_to_create_linda_problems_variant_one(self, previous_response_extension):
        message = [
            {"role": "system",
             "content": "Your task is to complete the last sentence of the following problem to construct a conjunction fallacy quiz. Do not mention the name 'conjunction fallacy'."
                        + self.random_roc_story + "\nWhich is more likely?\n(a) " + previous_response_extension + "\n(b) " + previous_response_extension[:-1] + " " + self.connector}
        ]
        return message

    def prompt_to_create_linda_problems_variant_one_irrelevant(self, previous_response_extension, previous_response_completion):
        message = [
            {"role": "system",
             "content": "Your task is to complete the last sentence of the following problem to construct a conjunction fallacy quiz. Do not mention the name 'conjunction fallacy'."
                        + self.random_roc_story + "\nWhich is more likely?\n(a) " + previous_response_extension + "\n(b) " + previous_response_extension[:-1] + " " + self.connector},
            {"role": "assistant",
             "content": previous_response_completion},
            {"role": "system",
             "content": "Your next task is to complete the last sentence of the same problem "
                        "but make sure your completion after '" + self.connector + "' is now irrelevant to the content intentionally: "
                        + self.random_roc_story + "\nWhich is more likely?\n(a) " + previous_response_extension + "\n(b) " + previous_response_extension[:-1] + " " + self.connector}
        ]
        return message


    ######## Prompts to create other Linda problems, variant three and four ########
    def prompt_to_create_linda_problems_variant_two(self):
        message = [
            {"role": "system",
             "content": "Your task is to complete the last sentence of the following problem to construct a conjunction fallacy quiz. Do not mention the name 'conjunction fallacy'."
                        + self.random_news_before_last_sentence + "\nWhich is more likely?\n(a) " + self.random_news_last_sentence + "\n(b) " + self.random_news_last_sentence[:-1] + " " + self.connector}
        ]
        return message

    def prompt_to_create_linda_problems_variant_two_irrelevant(self, previous_response_completion):
        message = [
            {"role": "system",
             "content": "Your task is to complete the last sentence of the following problem to construct a conjunction fallacy quiz. Do not mention the name 'conjunction fallacy'."
                        + self.random_news_before_last_sentence + "\nWhich is more likely?\n(a) " + self.random_news_last_sentence + "\n(b) " + self.random_news_last_sentence[:-1] + " " + self.connector},
            {"role": "assistant",
             "content": previous_response_completion},
            {"role": "system",
             "content": "Your next task is to complete the last sentence of the same problem "
                        "but make sure your completion after '" + self.connector + "' is now irrelevant to the content intentionally: "
                        + self.random_news_before_last_sentence + "\nWhich is more likely?\n(a) " + self.random_news_last_sentence + "\n(b) " + self.random_news_last_sentence[:-1] + " " + self.connector}
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
import torch
import random
from utils import *
from copy import deepcopy

class AllDataPrompts:
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
        elif self.linda_problem_variant == 'variant_three':
            self.all_disease_symptoms = load_disease_symptoms(args['datasets']['disease_symptoms_filename'])
            self.random_gender = 'female'
            self.random_age = 31
            self.random_race = 'asian'
        elif self.linda_problem_variant == 'variant_four':
            self.all_celebrity_names = load_celebrity_names(args['datasets']['celebrity_names_filename'])
        elif self.linda_problem_variant == 'variant_five':
            self.all_natural_disasters = load_natural_disasters(args['datasets']['natural_disasters_filename'])
            self.random_year = 2024
            self.random_gender = 'female'
            self.random_age = 31
            self.random_race = 'asian'

    def select_a_random_occupation(self):
        self.random_occupation = random.choice(self.all_occupations)

    def select_a_random_gender(self):
        self.random_gender = random.choice(['female', 'male', 'female', 'male', 'nonbinary', 'transgender', 'queer'])

    def select_a_random_age(self):
        if self.linda_problem_variant == 'variant_three':
            self.random_age = random.randint(40, 70)
        else:
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

    def select_a_random_disease_symptom_pair(self):
        self.random_disease_symptom_pair = self.all_disease_symptoms.sample(n=1)
        non_empty_columns = self.random_disease_symptom_pair.dropna(axis=1).columns.tolist()

        self.random_disease = self.random_disease_symptom_pair['Disease'].values[0]
        non_empty_columns.remove('Disease')
        self.random_symptoms = np.random.choice(non_empty_columns, size=2, replace=False)
        self.random_symptom_one, self.random_symptom_two = self.random_disease_symptom_pair[self.random_symptoms[0]].values[0], self.random_disease_symptom_pair[self.random_symptoms[1]].values[0]

    def select_a_random_celebrity(self):
        self.random_celebrity = random.choice(self.all_celebrity_names)

    def select_a_random_natural_disaster(self):
        self.random_disaster = random.choice(self.all_natural_disasters)

    def select_a_random_year(self):
        self.random_year = random.randint(2024, 2070)

    def select_random_letters(self):
        # smallest_index = random.randint(0, 2)
        # length = [random.randint(4, 8) for _ in range(3)]
        # length[smallest_index] = min(length) - 1    # random location for the one with the smallest length
        # _, _, self.letter1, self.letter2 = random_letter_pair_combination(length=min(length)-1)   # ensure one with the smallest length
        # self.random_letters = [random_letter_pair_combination(l, self.letter1, self.letter2)[0] for l in length] # fix the letter1 and letter2, only change lengths

        # ensure that all options have the same length
        length = random.randint(4, 8)
        _, _, self.letter1, self.letter2 = random_letter_pair_combination(length=length) # randomly pick two letters

        # randomly generate one sequences of letters
        self.random_letters = []
        letters, _, _, _ = random_letter_pair_combination(length, self.letter1, self.letter2)
        self.random_letters.append(deepcopy(letters))

        position = random.randint(0, length)
        self.random_letters.append(letters[:position] + self.letter1 + letters[position:])

        position = random.randint(0, length)
        self.random_letters.append(letters[:position] + self.letter2 + letters[position:])
        

        # when the unfair coin has more letter1, make sure the option with the most number of letter1 is correct
        self.random_letters_baseline = deepcopy(self.random_letters)
        letters = self.random_letters_baseline[0]
        indices = [i for i, x in enumerate(letters) if x == self.letter2]
        position = random.choice(indices)
        self.random_letters_baseline[0] = letters[:position] + self.letter1 + letters[position+1:] + self.letter1
        indices = [0,1,2]
        random.shuffle(indices)
        shuffled_dict = {idx: value for idx, value in zip([0,1,2], indices)}
        self.random_letters = [self.random_letters[shuffled_dict[i]] for i in [0,1,2]]
        self.random_letters_baseline = [self.random_letters_baseline[shuffled_dict[i]] for i in [0,1,2]]
        self.correct_option_baseline = self.correct_option = indices.index(0)


    def variant_six_suffix(self):
        return "Which sequence do you prefer to bet?\n" + \
               "(a) " + self.random_letters[0] + ".\n" + \
               "(b) " + self.random_letters[1] + ".\n" + \
               "(c) " + self.random_letters[2] + "."

    def variant_six_suffix_baseline(self):
        return "Which sequence do you prefer to bet?\n" + \
               "(a) " + self.random_letters_baseline[0] + ".\n" + \
               "(b) " + self.random_letters_baseline[1] + ".\n" + \
               "(c) " + self.random_letters_baseline[2] + "."


    ######## The original Linda problem ########
    def linda_problem(self):
        if self.linda_problem_variant == 'original':
            return "Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. " \
                   "As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations. " \
                   "Which is more probable?\n" \
                   "(a) Linda is a bank teller.\n" \
                   "(b) Linda is a bank teller and is active in the feminist movement."

        elif self.linda_problem_variant == 'variant_one' or self.linda_problem_variant == 'variant_two':    # these two variants have the same format but seed from different data sources
            return "John P. is a meek man, 42 years old, married with two children. His neighbors describe him as mild-mannered but somewhat secretive. " \
                   "He owns an import-export company based in New York City, and he travels frequently to Europe and the Far East. " \
                   "Mr. P. was convicted once for smuggling precious stones and metals (including uranium) and received a suspended sentence of 6 months in jail and a large fine. " \
                   "Mr. P. is currently under police investigation. " \
                   "Which one is more likely?\n" \
                   "(a) Mr. P. killed one of his employees.\n" \
                   "(b) Mr. P. killed one of his employees to prevent him from talking to the police."

        elif self.linda_problem_variant == 'variant_three':
            return "A 55-year-old woman had a pulmonary embolism (blood clot in the lung). Which one is more likely?\n" \
                   "(a) She also experiences Hemiparesis.\n" \
                   "(b) She also experiences Hemiparesis and Dyspnea."

        elif self.linda_problem_variant == 'variant_four':
            return "Suppose Bjorn Borg reaches the Wimbledon finals. Which outcome is more likely?\n" \
                   "(a) Borg will lose the first set.\n" \
                   "(b) Borg will lose the first set but win the match."

        elif self.linda_problem_variant == 'variant_five':
            return "Ricky, a meticulous risk manager based in California, had always been fascinated by the intricacies of natural disasters. " \
                   "Working amidst the stunning landscape of the Golden State, he found himself surrounded by the potential for earthquakes, wildfires, floods, and more. " \
                   "However, rather than being overwhelmed by fear, Ricky approached his job with a sense of curiosity and a drive to understand these phenomena better. " \
                   "Please tell Ricky which one is more likely to happen in 2024:\n" \
                   "(a) a massive flood somewhere in North America in 2024, in which more than 1000 people drown. " \
                   "(b) an earthquake in California sometime in 2024, causing a flood in which more than 1000 people drown."

        elif self.linda_problem_variant == 'variant_six':
            if self.args['datasets']['generate_mode'] == 'baseline':
                return "Consider a regular six-sided die with four green faces and two red faces. The die will be rolled 20 times and the sequence of greens (G) and reds (R) will be recorded. " \
                       "You are asked to select one sequence, from a set of three, and you will win $25 if the sequence you chose appears on successive rolls of the die. " \
                       "Which sequence do you prefer to bet?\n" \
                       "(a) RGRRR\n" \
                       "(b) GRGRRR\n" \
                       "(c) GRRRRR."
            else:
                return "Consider a regular six-sided die with four green faces and two red faces. The die will be rolled 20 times and the sequence of greens (G) and reds (R) will be recorded. " \
                       "You are asked to select one sequence, from a set of three, and you will win $25 if the sequence you chose appears on successive rolls of the die. "  # \


    ######## Prompts to create other Linda problems, baseline ########
    def prompt_to_create_linda_problems_baseline(self):
        message = [
            {"role": "system",
             "content": "Your task is to create another Linda problem following the example below.\n" + self.original_linda_problem + "\n"
                        "Ensure that your new problem maintains the same number of options, listing them as (a), (b), (c), etc. "
                        "Please write your example here:"}
        ]
        return message


    ######## Prompts to create other Linda problems, original version ########
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
                        "You shall include the question 'Which is more probable?' followed by two options (a) and (b), one of which should be a subset of the other."
                        "You can randomly switch the order of which option is (a) and which is (b). "
                        "The problem statement should exactly match the bio. "
                        "Use the employment occupation '" + self.random_occupation + "' for both options, "
                        "except that the hobby " + previous_response_hobby + " should also be included in the longer option only. "
                        "In the longer option, you can either put the hobby before or after the occupation. "
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
                        "You shall include the question 'Which is more probable?' followed by two options (a) and (b), one of which should be a subset of the other"
                        "You can randomly switch the order of which option is (a) and which is (b). "
                        "The problem statement should exactly match the bio. "
                        "Use the employment occupation '" + self.random_occupation + "' for both options, "
                        "except that the hobby " + previous_response_hobby + " should also be included in the longer option only. "
                        "In the longer option, you can either put the hobby before or after the occupation. "
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


    ######## Prompts to create other Linda problems, variant two ########
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


    ######## Prompts to create other Linda problems, variant three ########
    def prompt_to_create_linda_problems_variant_three(self):
        message = [
            {"role": "system",
             "content": "Your task is to create another conjunction fallacy quiz following the format in the example below. Do not mention the name 'conjunction fallacy'. Example:\n"
                        + self.original_linda_problem + "\n You should pick a random name for the patient, use gender " + self.random_gender + ", race " + self.random_race + ", an age " + str(self.random_age) +
                        "and the disease " + self.random_disease + "in your new problem statement. "
                        "The question should be 'Which one is more likely?' followed by two options (a) and (b), one of which should be a subset of the other. "
                        "You can randomly switch the order of which option is (a) and which is (b). "
                        "You should use the symptoms '" + self.random_symptom_one + "' in both options and add '" + self.random_symptom_two + "' to the longer option only. "
                        "Do not make any changes to the given disease or the symptoms.\n Here is the new problem:"}
        ]
        return message

    def prompt_to_create_linda_problems_variant_three_irrelevant(self):
        message = [
            {"role": "system",
             "content": "Your task is to create another conjunction fallacy quiz following the format in the example below. Do not mention the name 'conjunction fallacy'. Example:\n"
                        + self.original_linda_problem + "\n You should pick a random name for the patient, use gender " + self.random_gender + ", race " + self.random_race + ", an age " + str(self.random_age) +
                        "and the disease " + self.random_disease + "in your new problem statement. "
                        "The question should be 'Which one is more likely?' followed by two options (a) and (b), one of which should be a subset of the other. "
                        "You can randomly switch the order of which option is (a) and which is (b). "
                        "You should use the symptoms '" + self.random_symptom_one + "' in both options."
                        "You should add another random symptoms to the longer option only, which must be completely irrelevant to the disease " + self.random_disease + " intentionally. "
                        "Do not make any changes to the given disease or the symptoms.\n Here is the new problem:"}
        ]
        return message

    ######## Prompts to create other Linda problems, variant four ########

    def prompt_celebrity_few_shot(self):
        message = [
            {"role": "system",
             "content": 'Create one example that look like this:\n\n'
            'Suppose [celebrity is going to do something]. Which is more likely:\n'
            '(a) [Something unlikely for this person]\n'
            '(b) [Something unlikely for this person] but [something extremely likely for this person]\n\n'
            'Here are some examples:\n\n'
            'Suppose Taylor Swift is going to have another tour in 2027. Which is more likely:\n'
            '(a) Her first show is a flop.\n'
            '(b) Her first show is a flop but she will eventually sell over a million tickets for the entire tour.\n\n'
            'Suppose Joe Biden is running for president in 2024. Which is more likely:\n'
            '(a) Joe Biden will win the national popular vote\n'
            '(b) Joe Biden will win the national popular vote but lose the Electoral College vote\n\n'
            'Suppose Bjorn Borg reaches the Wimbledon finals. Which outcome is more likely?\n'
            '(a) Borg will lose the first set\n'
            '(b) Borg will lose the first set but win the match\n\n'
            'Complete the following. Do not output anything else.\n\n'
            f'Suppose {self.random_celebrity}'}
        ]
        return message
    
    def get_random_name_same_gender_as_celebrity(self, text):
        message = [
            {"role": "system",
             "content": f"Your task is to replace {self.random_celebrity} with a random first name of the same gender in the following text:\n\n{text}\n\n\nSimply write the result."
             }
        ]
        return message
        
    def parse_celebrity_few_shot(self, response, name = ''):
        # mode can be 'original' or 'random'
        # random mode is used for replacing the celebrity name with a random one
        if name == '':
            name = self.random_celebrity
        if f'Suppose {self.random_celebrity}' not in response:
            response = f'Suppose {name}{response} '
        else:
            response = response.replace(f'Suppose {self.random_celebrity}', f'Suppose {name}')
        return response
    '''
    def prompt_to_write_an_event(self):
        message = [
            {"role": "system",
             "content": "Your task is to write a possible event or situation the celebrity " + self.random_celebrity + " will be involved in using one sentence. "
                        "The event or situation must be related to their expertise or career. For example, 'Bjorn Borg reaches the Wimbledon finals."}
        ]
        return message

    def prompt_to_write_an_achievement(self, previous_response_event):
        message = [
            {"role": "system",
             "content": "Your task is to write a possible event or situation the celebrity " + self.random_celebrity + " will be involved in using one sentence. "
                        "The event or situation must be related to their expertise or career. For example, 'Bjorn Borg reaches the Wimbledon finals."},
            {"role": "assistant", "content": previous_response_event},
            {"role": "system",
             "content": "Your next task is to find a possible achievement of the same celebrity " + self.random_celebrity + " in that event or situation in one sentence. "
                        "For example, 'Bjorn Borg will win the match'. Avoid using words appeared in " + previous_response_event + " except for person's name."}
        ]
        return message

    def prompt_to_find_a_small_failure(self, previous_response_event):
        message = [
            {"role": "system",
             "content": "Your task is to write a possible event or situation the celebrity " + self.random_celebrity + " will be involved in using one sentence. "
                        "The event or situation must be related to their expertise or career. For example, 'Bjorn Borg reaches the Wimbledon finals."},
            {"role": "assistant", "content": previous_response_event},
            {"role": "user",
             "content": "Your next task is to think about a possible small failure of the celebrity " + self.random_celebrity + " in one sentence. "
                        "The small failure must be related to the event mentioned above. For example, 'Bjorn Borg will lose the first set'."
                        "Avoid using words appeared in " + previous_response_event + " except for person's name."}
        ]
        return message

    def prompt_to_create_linda_problems_variant_four(self, previous_response_event, previous_response_achievement, previous_response_failure):
        message = [
            {"role": "system",
             "content": "Your task is to write a possible event or situation the celebrity " + self.random_celebrity + " will be involved in using one sentence. "
                        "The event or situation must be related to their expertise or career. For example, 'Bjorn Borg reaches the Wimbledon finals."},
            {"role": "assistant", "content": previous_response_event},
            {"role": "system",
             "content": "Your next task is to find a possible achievement of the same celebrity " + self.random_celebrity + " in that event or situation in one sentence. "
                        "For example, 'Bjorn Borg will win the match'. Avoid using words appeared in " + previous_response_event + " except for person's name."},
            {"role": "assistant", "content": previous_response_achievement},
            {"role": "user",
             "content": "Your next task is to think about a possible small failure of the celebrity " + self.random_celebrity + " in one sentence. "
                        "The small failure must be related to the event mentioned above. For example, 'Bjorn Borg will lose the first set'."
                        "Avoid using words appeared in " + previous_response_event + " except for person's name."},
            {"role": "assistant", "content": previous_response_failure},
            {"role": "user",
             "content": "Your task is to summarize the achievement and small failure of the celebrity " + self.random_celebrity + " and create another conjunction fallacy quiz following the format in the example below. "
                        "Do not mention the name 'conjunction fallacy'. Example:\n" + self.original_linda_problem +
                        "\n The question should be 'Which outcome is more likely?' followed by two options (a) and (b), one of which should be a subset of the other. "
                        "You can randomly switch the order of which option is (a) and which is (b). "
                        "The problem statement should match the event mentioned in " + previous_response_event + ". "
                        "The two options should look identical and include the possible small failure '" + previous_response_failure + "', "
                        "except that you also should add the final achievement '" + previous_response_achievement + "' to the rest part of the longer option only. Avoid mentioning the event in the options. "
                        "\nHere is the new problem:"}
        ]
        return message

    def prompt_to_create_linda_problems_variant_four_irrelevant(self, previous_response_event, previous_response_achievement, previous_response_failure, previous_response_problem):
        message = [
            {"role": "system",
             "content": "Your task is to write a possible event or situation the celebrity " + self.random_celebrity + " will be involved in using one sentence. "
                        "The event or situation must be related to their expertise or career. For example, 'Bjorn Borg reaches the Wimbledon finals."},
            {"role": "assistant", "content": previous_response_event},
            {"role": "system",
             "content": "Your next task is to find a possible achievement of the same celebrity " + self.random_celebrity + " in that event or situation in one sentence. "
                        "For example, 'Bjorn Borg will win the match'. Avoid using words appeared in " + previous_response_event + " except for person's name."},
            {"role": "assistant", "content": previous_response_achievement},
            {"role": "user",
             "content": "Your next task is to think about a possible small failure of the celebrity " + self.random_celebrity + " in one sentence. "
                        "The small failure must be related to the event mentioned above. For example, 'Bjorn Borg will lose the first set'."
                        "Avoid using words appeared in " + previous_response_event + " except for person's name."},
            {"role": "assistant", "content": previous_response_failure},
            {"role": "system",
             "content": "Your task is to summarize the achievement and small failure of the celebrity " + self.random_celebrity + " and create another conjunction fallacy quiz following the format in the example below. "
                        "Do not mention the name 'conjunction fallacy'. Example:\n" + self.original_linda_problem +
                        "\n The question should be 'Which outcome is more likely?' followed by two options (a) and (b), one of which should be a subset of the other. "
                        "You can randomly switch the order of which option is (a) and which is (b). "
                        "The problem statement should match the event mentioned in " + previous_response_event + ". "
                        "The two options should look identical and include the possible small failure '" + previous_response_failure + "', "
                        "except that you also should add the final achievement '" + previous_response_achievement + "' to the rest part of the longer option only. Avoid mentioning the event in the options. "
                        "\nHere is the new problem:"},
            {"role": "assistant", "content": previous_response_problem},
            {"role": "user", "content": "Given the new problem you have generated, replace the final achievement in the longer option with a completely irrelevant achievement intentionally."
                                        "Do not mention " + previous_response_achievement + " anymore. Keep the rest of the problem statement the identical. Here is the new problem:"}
        ]
        return message

    
    def prompt_to_create_linda_problems_variant_four_nobody(self, previous_response_event, previous_response_achievement, previous_response_failure, previous_response_problem):
        message = [
            {"role": "system",
             "content": "Your task is to write a possible event or situation the celebrity " + self.random_celebrity + " will be involved in using one sentence. "
                        "The event or situation must be related to their expertise or career. For example, 'Bjorn Borg reaches the Wimbledon finals."},
            {"role": "assistant", "content": previous_response_event},
            {"role": "system",
             "content": "Your next task is to find a possible achievement of the same celebrity " + self.random_celebrity + " in that event or situation in one sentence. "
                        "For example, 'Bjorn Borg will win the match'. Avoid using words appeared in " + previous_response_event + " except for person's name."},
            {"role": "assistant", "content": previous_response_achievement},
            {"role": "user",
             "content": "Your next task is to think about a possible small failure of the celebrity " + self.random_celebrity + " in one sentence. "
                        "The small failure must be related to the event mentioned above. For example, 'Bjorn Borg will lose the first set'."
                        "Avoid using words appeared in " + previous_response_event + " except for person's name."},
            {"role": "assistant", "content": previous_response_failure},
            {"role": "system",
             "content": "Your task is to summarize the achievement and small failure of the celebrity " + self.random_celebrity + " and create another conjunction fallacy quiz following the format in the example below. "
                        "Do not mention the name 'conjunction fallacy'. Example:\n" + self.original_linda_problem +
                        "\n The question should be 'Which outcome is more likely?' followed by two options (a) and (b), one of which should be a subset of the other. "
                        "You can randomly switch the order of which option is (a) and which is (b). "
                        "The problem statement should match the event mentioned in " + previous_response_event + ". "
                        "The two options should look identical and include the possible small failure '" + previous_response_failure + "', "
                        "except that you also should add the final achievement '" + previous_response_achievement + "' to the rest part of the longer option only. Avoid mentioning the event in the options. "
                        "\nHere is the new problem:"},
            {"role": "assistant", "content": previous_response_problem},
            {"role": "user", "content": "Given the new problem you have generated, replace the celebrity name " + self.random_celebrity + " with a completely random name."
                                        "It should be the name of a nobody, not a celebrity, and you can use any frequent name in English. "
                                        "Keep the rest of the problem statement the identical. Here is the new problem:"}
        ]
        return message
    '''

    ######## Prompts to create other Linda problems, variant five ########
    def prompt_to_write_a_disaster(self):
        message = [
            {"role": "system",
             "content": "Your task is to describe a natural disaster '" + self.random_disaster + "' that could happen in the near future in one sentence. Do not mention the possible cause of the disaster."}
        ]
        return message

    def prompt_to_write_another_related_disaster(self, previous_response_disaster):
        message = [
            {"role": "system",
             "content": "Your task is to describe a natural disaster '" + self.random_disaster + "' that could happen in the near future in one sentence. Do not mention the possible cause of the disaster."},
            {"role": "assistant",
             "content": previous_response_disaster},
            {"role": "user",
             "content": "Your next task is to find another natural disaster that might be the cause of the previous one. This disaster should be able to trigger or cause " + self.random_disaster + ". "
                        "For example, if the previous one you picked was 'a massive flood somewhere in North America in 2024, in which more than 1000 people drown.', "
                        "you can now say 'an earthquake in California sometime in 2024, causing a massive flood somewhere in North America in 2024, in which more than 1000 people drown.'"
                        "Do not repeat this example. The new disaster should be written first, and then include '" + previous_response_disaster + "'as the rest of your answer."}
        ]
        return message

    def prompt_to_create_linda_problems_variant_five(self, previous_response_disaster, previous_response_disaster_related):
        message = [
            {"role": "system",
             "content": "Your task is to describe a natural disaster '" + self.random_disaster + "' that could happen in the near future in one sentence. Do not mention the possible cause of the disaster."},
            {"role": "assistant",
             "content": previous_response_disaster},
            {"role": "user",
             "content": "Your next task is to find another natural disaster that might be the cause of the previous one. This disaster should be able to trigger or cause " + self.random_disaster + ". "
                        "For example, if the previous one you picked was 'a massive flood somewhere in North America in 2024, in which more than 1000 people drown.', "
                        "you can now say 'an earthquake in California sometime in 2024, causing a massive flood somewhere in North America in 2024, in which more than 1000 people drown.'"
                        "Do not repeat this example. The new disaster should be written first, and then include '" + previous_response_disaster + "'as the rest of your answer."},
            {"role": "assistant",
             "content": previous_response_disaster_related},
            {"role": "user",
             "content": "You shall pick a random name, use gender " + self.random_gender + ", race " + self.random_race + ", and an age " + str(self.random_age) + ", "
                        "and write a short bio within 100 words. The bio should be related to the disasters '" + self.random_disaster + "' you have generated. "
                        "Summarize the information and create another conjunction fallacy quiz following the format in the example below. "
                        "Do not mention the name 'conjunction fallacy'. Example:\n" + self.original_linda_problem +
                        "\n The question should look like 'which one is more likely to happen?' followed by two options (a) and (b), one of which should be a subset of the other. "
                        "You can randomly switch the order of which option is (a) and which is (b). "
                        "Replace '2024' to " + str(self.random_year) + ". The problem statement should be the bio you have written. "
                        "The shorter option should match '" + previous_response_disaster + "', and the longer option should match '" + previous_response_disaster_related + "'. "
                        "\nHere is the new problem:"}
        ]
        return message


    def prompt_to_create_linda_problems_variant_five_irrelevant(self, previous_response_disaster, previous_response_disaster_related, previous_response_problem):
        message = [
            {"role": "system",
             "content": "Your task is to describe a natural disaster '" + self.random_disaster + "' that could happen in the near future in one sentence. Do not mention the possible cause of the disaster."},
            {"role": "assistant",
             "content": previous_response_disaster},
            {"role": "user",
             "content": "Your next task is to find another natural disaster that might be the cause of the previous one. This disaster should be able to trigger or cause " + self.random_disaster + ". "
                        "For example, if the previous one you picked was 'a massive flood somewhere in North America in 2024, in which more than 1000 people drown.', "
                        "you can now say 'an earthquake in California sometime in 2024, causing a massive flood somewhere in North America in 2024, in which more than 1000 people drown.'"
                        "Do not repeat this example. The new disaster should be written first, and then include '" + previous_response_disaster + "'as the rest of your answer."},
            {"role": "assistant",
             "content": previous_response_disaster_related},
            {"role": "user",
             "content": "You shall pick a random name, use gender " + self.random_gender + ", race " + self.random_race + ", and an age " + str(self.random_age) + ", "
                        "and write a short bio within 100 words. The bio should be related to the disasters '" + self.random_disaster + "' you have generated. "
                        "Summarize the information and create another conjunction fallacy quiz following the format in the example below. "
                        "Do not mention the name 'conjunction fallacy'. Example:\n" + self.original_linda_problem +
                        "\n The question should look like 'which one is more likely to happen?' followed by two options (a) and (b), one of which should be a subset of the other. "
                        "You can randomly switch the order of which option is (a) and which is (b). "
                        "Replace '2024' to " + str(self.random_year) + ". The problem statement should be the bio you have written. "
                        "The shorter option should exactly match '" + previous_response_disaster + "', and the longer option should exactly match '" + previous_response_disaster_related + "'. "
                        "\nHere is the new problem:"},
            {"role": "assistant",
             "content": previous_response_problem},
            {"role": "user",
             "content": "Given the new problem you have generated, in the longer option '" + previous_response_disaster_related + "' there is one disaster that causes the other."
                        "Replace the disaster that causes the other with a completely irrelevant disaster to break their 'causing' relationship intentionally. "
                        "Keep the rest of the problem statement the identical. Here is the new problem:"}
        ]
        return message


    ######## Prompts to create other Linda problems, variant six ########
    def prompt_to_create_linda_problems_variant_six(self):
        message = [
            {"role": "system",
             "content": "Your task is to create a new problem following the example below.\n" + self.original_linda_problem + "\n + "
                        f"You should change the two colors mentioned in the example to {color_dict[self.letter1]} and {color_dict[self.letter2]}."                                                                                                      
                        "You can modify the die to any other object with different colors and numbers of faces, or change the prize value. Just make sure that the letters will match."
                        "However, always make sure that the die or any other object is unfair and has MORE " + self.letter1 + " than " + self.letter2 + " in your new problem. "
                        "Do NOT add any options or sequences to the problem at this moment."
                        "\nHere is the new problem:"}
        ]
        return message

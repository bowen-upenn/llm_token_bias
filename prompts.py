######## The original Linda problem ########
def linda_problem():
    return "Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. " \
           "As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations. " \
           "Which is more probable?\n" \
           "(a) Linda is a bank teller.\n" \
           "(b) Linda is a bank teller and is active in the feminist movement."


######## Prompts to create other Linda problems into a synthetic dataset ########
def prompt_to_create_linda_problems_baseline():
    original_linda_problem = linda_problem()
    message = [
        {"role": "system",
         "content": "Your task is to create another Linda problem following the example below.\n" + original_linda_problem + "\n"
                    "Write another example here:"}
    ]
    return message


def prompt_to_write_a_bio():
    message = [
        {"role": "system",
         "content": "Your task is to write a short bio for a random person within 100 words. "
                    "You shall pick a random name in English, age between 25 and 50, and use gender pronouns 'she', 'he', or 'they'. "
                    "The bio should describe the college majors, some personal characters, and interests. Keep the bio short.\n"
                    "For example, 'Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. "
                    "As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations.'\n"
                    "Write another example here:"}
    ]
    return message


def prompt_to_find_a_hobby(previous_response):
    message = [
        {"role": "system",
         "content": "Your task is to write a short bio for a random person within 100 words. "
                    "You shall pick a random name in English, age between 20 and 50, and use gender pronouns 'she', 'he', or 'they'. "
                    "The bio should describe the college majors, some personal characters, and interests. Keep the bio short.\n"
                    "For example, 'Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. "
                    "As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations.'\n"
                    "Write another example here:"},
        {"role": "assistant", "content": previous_response},
        {"role": "user",
         "content": "Your next step is to find a hobby or activity that the person will be interested in based on your experience. "
                    "The hobby or activity must be relevant to the bio descriptions. "
                    "In the example above, we can say that 'Linda is active in the feminist movement.' "
                    "Please keep your answer in one sentence and begin with that person's name, like 'Linda is', "
                    "but refrain from using any words used in the bio."}
    ]
    return message


def prompt_to_create_golden_linda_problems(previous_response_bio, previous_response_hobby):
    original_linda_problem = linda_problem()
    message = [
        {"role": "system",
         "content": "Your task is to write a short bio for a random person within 100 words. "
                    "You shall pick a random name in English, age between 20 and 50, and use gender pronouns 'she', 'he', or 'they'. "
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
         "content": "Here is an example of a complete Linda problem: " + original_linda_problem + "\n"
                    "Your final step is to create another Linda problem based on the bio and hobby you have written. "
                    "The problem should have two options, one of which is a subset of the other, and you can choose a random employment occupation irrelevant to the bio. "
                    "Please write another example here:"}
    ]
    return message


######## Prompts to evaluate the model's ability in answering the problems in the synthetic dataset ########
def prompt_to_answer_the_question(question):
    message = [
        {"role": "system", "content": ""},  # add instruction here
        {"role": "user", "content": ""}  # add question here
    ]
    return message


def prompt_to_critic_the_answer(question, model_answer):
    message = [
        {"role": "system", "content": ""},  # add instruction and question here
        {"role": "user", "content": ""}  # add model response here
    ]  # Note: ask the model to attach special tokens in the response, such as [Yes] or [No]
    return message


def prompt_to_reanswer_the_question(question, init_model_answer, critic):
    message = [
        {"role": "system", "content": ""},  # add initial instruction here
        {"role": "user", "content": ""},  # add initial question here
        {"role": "system", "content": ""},  # add initial model response here
        {"role": "user", "content": ""},  # add critic here
        {"role": "system", "content": ""}  # add reattempt instruction here
    ]
    return message


def prompt_to_grade_the_answer(question, target_answer, model_answer, grader_id=0):
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


def prompt_to_generate_synthetic_data(question):
    message = [
        {"role": "system", "content": ""},  # add instruction here
        {"role": "user", "content": ""}  # add in-context learning example here
    ]
    return message
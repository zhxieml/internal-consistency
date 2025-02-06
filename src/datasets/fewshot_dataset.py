""" Format the dataset to be used in the model."""

import ast
import json
import random
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils import set_seed

ANSWER_TRIGGER = "So the answer is"


def setup_dataset(dataset, path, least_to_most=False):
    if dataset == "answer":
        if least_to_most:
            print(
                "Warning: least_to_most is not supported for answer dataset. Ignoring."
            )
        ds = load_answer_file(path)
    elif dataset.startswith("prontoqa"):
        ds = load_prontoqa(path, least_to_most=least_to_most)
    elif dataset.startswith("proofwriter"):
        ds = load_proofwriter(path, least_to_most=least_to_most)
    elif dataset.startswith("boolq"):
        ds = load_boolq(path, least_to_most=least_to_most)
    elif dataset.startswith("coinflip"):
        ds = load_coin_flip(path, least_to_most=least_to_most)
    elif dataset.startswith("swag"):
        ds = load_swag(path, least_to_most=least_to_most)
    elif dataset.startswith("mathqa"):
        ds = load_mathqa(path, least_to_most=least_to_most)
    else:
        raise NotImplementedError

    return ds


@dataclass
class Example:
    """Class to represent an example."""

    context: str
    question: str  # contains the query
    chain_of_thought: List[str]
    answer: str

    dict = asdict


@dataclass
class Sample:
    """Class to represent a sample."""

    examples: List[Example]
    context: str
    question: str  # contains the query
    chain_of_thought: List[str]
    answer: str
    model_output: Optional[str] = None

    dict = asdict


def read_jsonl(path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


class FewshotDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        super().__init__()
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return str(self.samples[index])


def setup_data_loader(
    dataset,
    seed=42,
    batch_size=8,
    collate_fn=lambda batch: [ast.literal_eval(i) for i in batch],
    shuffle=False,
):
    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    set_seed(seed)
    worker_seed = torch.initial_seed() % 2**32
    print("worker_seed : {}".format(worker_seed))

    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(worker_seed)

    dataloader = DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=False,
        num_workers=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return dataloader


def format_icl(
    sample,
    use_cot=True,
    use_icl=True,
    use_context=True,
    answer_formatter=lambda ans: f"{ANSWER_TRIGGER} {ans}",
):
    """Prompt engineering.

    `
    {icl_examples}

    {context}
    Q: {question}
    A: {chain_of_thought} {answer}
    `

    NOTE: {icl_examples}, {chain_of_thought}, and {answer} are optional
    """
    prompt = ""

    # format the in-context examples
    if use_icl:
        examples = sample["examples"]
        for i, example in enumerate(examples):
            context = example["context"]
            question = example["question"]
            chain_of_thought = example["chain_of_thought"]
            answer = example["answer"]

            if use_context and context is not None:
                query_prompt = f"{context}\nQ: {question}"
            else:
                query_prompt = f"Q: {question}"
            if use_cot and len(chain_of_thought) > 0:
                answer_prompt = (
                    f"A: {' '.join(chain_of_thought)} {answer_formatter(answer)}"
                )
            else:
                answer_prompt = f"A: {answer_formatter(answer)}"
            prompt += query_prompt + "\n"
            prompt += answer_prompt + "\n\n"

    # format the test example
    context = sample["context"]
    question = sample["question"]
    chain_of_thought = sample["chain_of_thought"]
    answer = sample["answer"]

    if use_context and context is not None:
        query_prompt = f"{context}\nQ: {question}"
    else:
        query_prompt = f"Q: {question}"
    answer_prompt = "A:"
    prompt += query_prompt + "\n"
    prompt += answer_prompt

    return prompt


def load_prontoqa(path, least_to_most=False):
    raw_data = json.load(open(path))
    samples = []

    for instance in raw_data.values():
        examples = []

        if not least_to_most:
            for k, v in instance.items():
                if k.startswith("in_context_example"):
                    example = Example(
                        context=v["question"],
                        question=v["query"],
                        chain_of_thought=v["chain_of_thought"],
                        answer=v["answer"],
                    )
                    examples.append(example.dict())
        else:
            examples.append(
                Example(
                    context="Every tumpus is sour. Tumpuses are rompuses. Every rompus is small. Every rompus is an impus. Impuses are floral. Impuses are dumpuses. Dumpuses are not kind. Every dumpus is a yumpus. Each yumpus is feisty. Yumpuses are numpuses. Numpuses are not opaque. Each numpus is a zumpus. Every wumpus is opaque. Every zumpus is temperate. Each zumpus is a jompus. Every jompus is dull. Jompuses are vumpuses. Alex is a dumpus.",
                    question="True or false: Alex is not opaque.",
                    chain_of_thought=[
                        "To answer this question, we need to know: 1. What are the properties of a dumpus? 2. Is Alex a dumpus? 3. Is a dumpus opaque?",
                        "1. Dumpuses are not kind, feisty, and yumpuses.",
                        "2. Alex is a dumpus.",
                        "3. Numpuses (which include dumpuses) are not opaque.",
                    ],
                    answer="True",
                )
            )

            examples.append(
                Example(
                    context="Every jompus is opaque. Jompuses are numpuses. Each numpus is bright. Numpuses are vumpuses. Each vumpus is not blue. Vumpuses are dumpuses. Dumpuses are hot. Dumpuses are tumpuses. Tumpuses are kind. Every tumpus is a wumpus. Every wumpus is happy. Wumpuses are rompuses. Each rompus is metallic. Each rompus is a yumpus. Each yumpus is not small. Zumpuses are small. Yumpuses are impuses. Max is a wumpus.",
                    question="True or false: Max is small.",
                    chain_of_thought=[
                        "To answer this question, we need to know: 1. What are the properties of a wumpus? 2. Is Max a wumpus? 3. Are wumpuses small?",
                        "1. Wumpuses are happy and rompuses.",
                        "2. Max is a wumpus.",
                        "3. Rompuses (which include wumpuses) are metallic and yumpuses. Yumpuses are not small.",
                    ],
                    answer="False",
                )
            )

            examples.append(
                Example(
                    context="Tumpuses are not angry. Tumpuses are wumpuses. Wumpuses are happy. Every wumpus is a dumpus. Dumpuses are temperate. Each dumpus is an impus. Each impus is not spicy. Impuses are yumpuses. Yumpuses are not fruity. Each jompus is fruity. Each yumpus is a vumpus. Each vumpus is bright. Every vumpus is a numpus. Wren is a dumpus.",
                    question="True or false: Wren is not fruity.",
                    chain_of_thought=[
                        "To answer this question, we need to know: 1. What are the properties of a dumpus? 2. Is Wren a dumpus? 3. Are dumpuses fruity?",
                        "1. Dumpuses are temperate and impuses.",
                        "2. Wren is a dumpus.",
                        "3. Yumpuses (which include dumpuses) are not fruity.",
                    ],
                    answer="True",
                )
            )

            examples.append(
                Example(
                    context="Every dumpus is large. Dumpuses are yumpuses. Every yumpus is amenable. Each yumpus is a jompus. Jompuses are opaque. Each jompus is a numpus. Each numpus is bright. Numpuses are zumpuses. Every zumpus is not luminous. Every zumpus is a rompus. Rompuses are not shy. Rompuses are impuses. Every impus is not cold. Impuses are wumpuses. Wumpuses are not spicy. Each wumpus is a tumpus. Vumpuses are luminous. Max is a jompus.",
                    question="True or false: Max is luminous.",
                    chain_of_thought=[
                        "To answer this question, we need to know: 1. What are the properties of a jompus? 2. Is Max a jompus? 3. Are jompuses luminous?",
                        "1. Jompuses are opaque and numpuses.",
                        "2. Max is a jompus.",
                        "3. Zumpuses (which include numpuses and jompuses) are not luminous.",
                    ],
                    answer="False",
                )
            )

            examples = [e.dict() for e in examples]

        sample = Sample(
            examples=examples,
            context=instance["test_example"]["question"],
            question=instance["test_example"]["query"],
            chain_of_thought=instance["test_example"]["chain_of_thought"],
            answer=instance["test_example"]["answer"],
        )
        samples.append(sample.dict())

    return FewshotDataset(samples)


def load_proofwriter(path, least_to_most=False):
    raw_data = json.load(open(path))
    samples = []

    for instance in raw_data.values():
        examples = []

        if not least_to_most:
            for k, v in instance.items():
                if k.startswith("in_context_example"):
                    example = Example(
                        context=v["question"],
                        question=v["query"],
                        chain_of_thought=v["chain_of_thought"],
                        answer=v["answer"],
                    )
                    examples.append(example.dict())
        else:
            examples.append(
                Example(
                    context="Fiona is blue. Harry is cold. Harry is white. If Harry is blue and Harry is green then Harry is not round. If something is green then it is young. All white things are young. If something is green and not white then it is not blue. Young, round things are not furry. If something is white and young then it is round. If something is young and not cold then it is round. If something is green and not young then it is not round.",
                    question="True or false: Harry is round.",
                    chain_of_thought=[
                        "To answer this question, we need to know: 1. What are Harry's characteristics? 2. What does being white imply about being young? 3. What does being white and young imply about being round?",
                        "1. Harry is cold and white.",
                        "2. All white things are young. Therefore, Harry is young.",
                        "3. If something is white and young then it is round. Therefore, Harry is round.",
                    ],
                    answer="True",
                )
            )

            examples.append(
                Example(
                    context="Charlie is big. Charlie is cold. Charlie is green. Charlie is rough. Charlie is young. Erin is big. Erin is white. Erin is young. Gary is big. Gary is green. Gary is rough. Gary is white. Harry is cold. Harry is white. Harry is young. All cold things are furry. All young, big things are furry. White things are rough. All rough things are big. Furry, cold things are young. If something is green then it is cold. If something is big and rough then it is green. Rough, big things are young.",
                    question="True or false: Erin is not green.",
                    chain_of_thought=[
                        "To answer this question, we need to know: 1. What are Erin's characteristics? 2. What does being white imply about being rough? 3. What does being rough imply about being green?",
                        "1. Erin is big, white, and young.",
                        "2. White things are rough. Therefore, Erin is rough.",
                        "3. If something is big and rough then it is green. Therefore, Erin is green.",
                    ],
                    answer="False",
                )
            )

            examples.append(
                Example(
                    context="The bear eats the mouse. The bear is kind. The bear is round. The bear is young. The bear likes the mouse. The mouse chases the bear. The mouse is green. The mouse is kind. The mouse is round. The mouse likes the bear. If something likes the mouse then it chases the bear. If something is kind and it chases the bear then it chases the mouse. If something chases the mouse and the mouse chases the bear then it eats the bear.",
                    question="True or false: The bear chases the mouse.",
                    chain_of_thought=[
                        "To answer this question, we need to know: 1. What are the interactions between the bear and the mouse? 2. Does the bear chase the mouse if it is kind and chases the bear?",
                        "1. The bear likes the mouse, so it chases the bear.",
                        "2. The bear is kind and it chases the bear, so it chases the mouse.",
                    ],
                    answer="True",
                )
            )

            examples.append(
                Example(
                    context="Erin is kind. Fiona is rough. All green, smart things are rough. All round things are young. Rough things are green. Smart, rough things are kind. All green things are round. Kind things are rough.",
                    question="True or false: Erin is green.",
                    chain_of_thought=[
                        "To answer this question, we need to know: 1. What does being kind imply about being rough? 2. What does being rough imply about being green?",
                        "1. Erin is kind.",
                        "2. Kind things are rough. Therefore, Erin is rough.",
                        "3. Rough things are green. Therefore, Erin is green.",
                    ],
                    answer="True",
                )
            )

            examples = [e.dict() for e in examples]

        sample = Sample(
            examples=examples,
            context=instance["test_example"]["question"],
            question=instance["test_example"]["query"],
            chain_of_thought=instance["test_example"]["chain_of_thought"],
            answer=instance["test_example"]["answer"],
        )
        samples.append(sample.dict())

    return FewshotDataset(samples)


def load_answer_file(path):
    raw_data = read_jsonl(path)
    samples = []

    for instance in raw_data:
        examples = [Example(**e) for e in instance["sample"]["examples"]]
        context = instance["sample"]["context"]
        question = instance["sample"]["question"]
        answer = instance["sample"]["answer"]
        chain_of_thought = instance["sample"]["chain_of_thought"]
        model_output = instance["model_output"]

        if isinstance(model_output, list):
            for out in model_output:
                sample = Sample(
                    examples=examples,
                    context=context,
                    question=question,
                    chain_of_thought=chain_of_thought,
                    answer=answer,
                    model_output=out,
                )
                samples.append(sample.dict())
        else:
            sample = Sample(
                examples=examples,
                context=context,
                question=question,
                chain_of_thought=chain_of_thought,
                answer=answer,
                model_output=model_output,
            )
            samples.append(sample.dict())

    return FewshotDataset(samples)


def load_answer_file_splitall(path, use_gt_cot=False):
    raw_data = read_jsonl(path)
    samples = []

    for instance in raw_data:
        examples = [Example(**e) for e in instance["sample"]["examples"]]
        question = instance["sample"]["question"]
        answer = instance["sample"]["answer"]
        samples.append(
            Sample(
                examples=examples,
                question=question,
                chain_of_thought=[""],
                answer=answer,
            ).dict()
        )

        if use_gt_cot:
            chain_of_thought = instance["sample"]["chain_of_thought"]
            for step in range(len(chain_of_thought)):
                cot = " ".join(chain_of_thought[: step + 1])
                sample = Sample(
                    examples=examples,
                    question=question,
                    chain_of_thought=[cot],
                    answer=answer,
                )
                samples.append(sample.dict())
        else:
            chain_of_thought = instance["model_output"].split(ANSWER_TRIGGER)[0].strip()
            chain_of_thought_split = chain_of_thought.split(". ")
            for step in range(len(chain_of_thought_split)):
                cot = ". ".join(chain_of_thought_split[: step + 1])
                cot += "." if not cot.endswith(".") else ""
                sample = Sample(
                    examples=examples,
                    question=question,
                    chain_of_thought=[cot],
                    answer=answer,
                )
                samples.append(sample.dict())

    return FewshotDataset(samples)


def process_boolq(instance):
    question = instance["question"].strip()
    question = question[0].upper() + question[1:] + "?"
    question = "True or false: " + question
    answer = "True" if instance["answer"] else "False"
    chain_of_thought = []
    context = instance["passage"].strip()

    return context, question, chain_of_thought, answer


def create_boolq_examples(least_to_most=False):
    examples = []

    if not least_to_most:
        examples.append(
            Example(
                context="System of a Down, sometimes shortened to System and abbreviated as SOAD, is an Armenian-American heavy metal band from Glendale, California, formed in 1994. The band currently consists of Serj Tankian (lead vocals, keyboards), Daron Malakian (vocals, guitar), Shavo Odadjian (bass, backing vocals) and John Dolmayan (drums).",
                question="True or false: Does system of a down have 2 singers?",
                chain_of_thought=[
                    "System of a Down currently consists of Serj Tankian, Daron Malakian, Shavo Odadjian and John Dolmayan. Serj and Daron do vocals, so the band does have two singers."
                ],
                answer="True",
            )
        )
        examples.append(
            Example(
                context="Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet.",
                question="True or false: Do iran and afghanistan speak the same language?",
                chain_of_thought=[
                    "Iran and Afghanistan both speak the Indo-European language Persian."
                ],
                answer="True",
            )
        )
        examples.append(
            Example(
                context="Both the violin and viola are played under the jaw. The viola, being the larger of the two instruments, has a playing range that reaches a perfect fifth below the violin's. The cello is played sitting down with the instrument between the knees, and its playing range reaches an octave below the viola's. The double bass is played standing or sitting on a stool, with a range that typically reaches a minor sixth, an octave or a ninth below the cello's.",
                question="True or false: Is a cello and a bass the same thing?",
                chain_of_thought=[
                    "The cello is played sitting down with the instrument between the knees, whereas the double bass is played standing or sitting on a stool."
                ],
                answer="False",
            )
        ),
        examples.append(
            Example(
                context="Epsom railway station serves the town of Epsom in Surrey. It is located off Waterloo Road and is less than two minutes' walk from the High Street. It is not in the London Oyster card zone unlike Epsom Downs or Tattenham Corner stations. The station building was replaced in 2012/2013 with a new building with apartments above the station (see end of article).",
                question="True or false: Can you use oyster card at epsom station?",
                chain_of_thought=[
                    "Epsom railway station serves the town of Epsom in Surrey and is not in the London Oyster card zone."
                ],
                answer="False",
            )
        )

    else:
        examples.append(
            Example(
                context="System of a Down, sometimes shortened to System and abbreviated as SOAD, is an Armenian-American heavy metal band from Glendale, California, formed in 1994. The band currently consists of Serj Tankian (lead vocals, keyboards), Daron Malakian (vocals, guitar), Shavo Odadjian (bass, backing vocals) and John Dolmayan (drums).",
                question="True or false: Does system of a down have 2 singers?",
                chain_of_thought=[
                    "To answer this question, we need to know: 1. Who are the members of System of a Down? 2. What are the roles of each member? 3. How many members are listed as vocalists?",
                    "1. The members are Serj Tankian, Daron Malakian, Shavo Odadjian, and John Dolmayan.",
                    "2. Serj Tankian is the lead vocalist, and Daron Malakian is also a vocalist.",
                    "3. There are two vocalists.",
                ],
                answer="True",
            )
        )

        examples.append(
            Example(
                context="Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet.",
                question="True or false: Do iran and afghanistan speak the same language?",
                chain_of_thought=[
                    "To answer this question, we need to know: 1. What is the primary language spoken in Iran? 2. What is the primary language spoken in Afghanistan? 3. Are these languages the same?",
                    "1. The primary language spoken in Iran is Persian (Farsi).",
                    "2. The primary language spoken in Afghanistan is also Persian (known as Dari).",
                    "3. Both countries primarily speak Persian.",
                ],
                answer="True",
            )
        )

        examples.append(
            Example(
                context="Both the violin and viola are played under the jaw. The viola, being the larger of the two instruments, has a playing range that reaches a perfect fifth below the violin's. The cello is played sitting down with the instrument between the knees, and its playing range reaches an octave below the viola's. The double bass is played standing or sitting on a stool, with a range that typically reaches a minor sixth, an octave or a ninth below the cello's.",
                question="True or false: Is a cello and a bass the same thing?",
                chain_of_thought=[
                    "To answer this question, we need to know: 1. What is the playing position of the cello? 2. What is the playing position of the double bass? 3. Are these playing positions the same?",
                    "1. The cello is played sitting down with the instrument between the knees.",
                    "2. The double bass is played standing or sitting on a stool.",
                    "3. The playing positions are different.",
                ],
                answer="False",
            )
        )

        examples.append(
            Example(
                context="Epsom railway station serves the town of Epsom in Surrey. It is located off Waterloo Road and is less than two minutes' walk from the High Street. It is not in the London Oyster card zone unlike Epsom Downs or Tattenham Corner stations. The station building was replaced in 2012/2013 with a new building with apartments above the station (see end of article).",
                question="True or false: Can you use oyster card at epsom station?",
                chain_of_thought=[
                    "To answer this question, we need to know: 1. Where is Epsom railway station located? 2. Is Epsom railway station in the London Oyster card zone?",
                    "1. Epsom railway station is in Surrey.",
                    "2. Epsom railway station is not in the London Oyster card zone.",
                ],
                answer="False",
            )
        )

    return examples


def load_boolq(path, least_to_most=False):
    raw_data = read_jsonl(path)
    samples = []
    examples = create_boolq_examples(least_to_most=least_to_most)

    for instance in raw_data:
        context, question, chain_of_thought, answer = process_boolq(instance)

        sample = Sample(
            context=context,
            examples=examples,
            question=question,
            chain_of_thought=chain_of_thought,
            answer=answer,
        )
        samples.append(sample.dict())

    return FewshotDataset(samples)


def create_coin_flip_examples(least_to_most=False):
    examples = []
    question = "True or false: Is the coin still heads up?"

    if not least_to_most:
        examples.append(
            Example(
                context="A coin is heads up. Ka flips the coin. Sherrie flips the coin.",
                question=question,
                chain_of_thought=[
                    "The coin was flipped by Ka and Sherrie. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up."
                ],
                answer="True",
            )
        )
        examples.append(
            Example(
                context="A coin is heads up. Jamey flips the coin. Teressa flips the coin.",
                question=question,
                chain_of_thought=[
                    "The coin was flipped by Jamey and Teressa. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up."
                ],
                answer="True",
            )
        )
        examples.append(
            Example(
                context="A coin is heads up. Maybelle flips the coin. Shalonda does not flip the coin.",
                question=question,
                chain_of_thought=[
                    "The coin was flipped by Maybelle. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up."
                ],
                answer="False",
            )
        )
        examples.append(
            Example(
                context="A coin is heads up. Millicent does not flip the coin. Conception flips the coin.",
                question=question,
                chain_of_thought=[
                    "The coin was flipped by Conception. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up."
                ],
                answer="False",
            )
        )
        examples.append(
            Example(
                context="A coin is heads up. Sal flips the coin. Raymond does not flip the coin.",
                question=question,
                chain_of_thought=[
                    "The coin was flipped by Sal. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up."
                ],
                answer="False",
            )
        )
        examples.append(
            Example(
                context="A coin is heads up. Conception flips the coin. Kristian does not flip the coin.",
                question=question,
                chain_of_thought=[
                    "The coin was flipped by Conception. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up."
                ],
                answer="False",
            )
        )
        examples.append(
            Example(
                context="A coin is heads up. Inga does not flip the coin. Elanor does not flip the coin.",
                question=question,
                chain_of_thought=[
                    "The coin was flipped by no one. So the coin was flipped 0 times. The coin started heads up, and it was not flipped, so it is still heads up."
                ],
                answer="True",
            )
        )
        examples.append(
            Example(
                context="A coin is heads up. Ryan flips the coin. Shaunda flips the coin.",
                question=question,
                chain_of_thought=[
                    "The coin was flipped by Ryan and Shaunda. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up."
                ],
                answer="True",
            )
        )

    else:
        examples.append(
            Example(
                context="A coin is heads up. Ka flips the coin. Sherrie flips the coin.",
                question="True or false: The coin is heads up after both flips?",
                chain_of_thought=[
                    "To answer this question, we need to know: 1. What is the starting position of the coin? 2. What happens to the coin after each flip?",
                    "1. The coin starts heads up.",
                    "2. Ka flips the coin, changing it to tails up.",
                    "3. Sherrie flips the coin, changing it back to heads up.",
                ],
                answer="True",
            )
        )

        examples.append(
            Example(
                context="A coin is heads up. Jamey flips the coin. Teressa flips the coin.",
                question="True or false: The coin is heads up after both flips?",
                chain_of_thought=[
                    "To answer this question, we need to know: 1. What is the starting position of the coin? 2. What happens to the coin after each flip?",
                    "1. The coin starts heads up.",
                    "2. Jamey flips the coin, changing it to tails up.",
                    "3. Teressa flips the coin, changing it back to heads up.",
                ],
                answer="True",
            )
        )

        examples.append(
            Example(
                context="A coin is heads up. Maybelle flips the coin. Shalonda does not flip the coin.",
                question="True or false: The coin is heads up after both flips?",
                chain_of_thought=[
                    "To answer this question, we need to know: 1. What is the starting position of the coin? 2. What happens to the coin after each flip?",
                    "1. The coin starts heads up.",
                    "2. Maybelle flips the coin, changing it to tails up.",
                    "3. Shalonda does not flip the coin, so it remains tails up.",
                ],
                answer="False",
            )
        )

        examples.append(
            Example(
                context="A coin is heads up. Millicent does not flip the coin. Conception flips the coin.",
                question="True or false: The coin is heads up after both flips?",
                chain_of_thought=[
                    "To answer this question, we need to know: 1. What is the starting position of the coin? 2. What happens to the coin after each flip?",
                    "1. The coin starts heads up.",
                    "2. Millicent does not flip the coin, so it remains heads up.",
                    "3. Conception flips the coin, changing it to tails up.",
                ],
                answer="False",
            )
        )

        examples.append(
            Example(
                context="A coin is heads up. Sal flips the coin. Raymond does not flip the coin.",
                question="True or false: The coin is heads up after both flips?",
                chain_of_thought=[
                    "To answer this question, we need to know: 1. What is the starting position of the coin? 2. What happens to the coin after each flip?",
                    "1. The coin starts heads up.",
                    "2. Sal flips the coin, changing it to tails up.",
                    "3. Raymond does not flip the coin, so it remains tails up.",
                ],
                answer="False",
            )
        )

        examples.append(
            Example(
                context="A coin is heads up. Conception flips the coin. Kristian does not flip the coin.",
                question="True or false: The coin is heads up after both flips?",
                chain_of_thought=[
                    "To answer this question, we need to know: 1. What is the starting position of the coin? 2. What happens to the coin after each flip?",
                    "1. The coin starts heads up.",
                    "2. Conception flips the coin, changing it to tails up.",
                    "3. Kristian does not flip the coin, so it remains tails up.",
                ],
                answer="False",
            )
        )

        examples.append(
            Example(
                context="A coin is heads up. Inga does not flip the coin. Elanor does not flip the coin.",
                question="True or false: The coin is heads up after both flips?",
                chain_of_thought=[
                    "To answer this question, we need to know: 1. What is the starting position of the coin? 2. What happens to the coin after each flip?",
                    "1. The coin starts heads up.",
                    "2. Inga does not flip the coin, so it remains heads up.",
                    "3. Elanor does not flip the coin, so it remains heads up.",
                ],
                answer="True",
            )
        )

        examples.append(
            Example(
                context="A coin is heads up. Ryan flips the coin. Shaunda flips the coin.",
                question="True or false: The coin is heads up after both flips?",
                chain_of_thought=[
                    "To answer this question, we need to know: 1. What is the starting position of the coin? 2. What happens to the coin after each flip?",
                    "1. The coin starts heads up.",
                    "2. Ryan flips the coin, changing it to tails up.",
                    "3. Shaunda flips the coin, changing it back to heads up.",
                ],
                answer="True",
            )
        )

    return examples


def load_coin_flip(path, least_to_most=False):
    with open(path, "r") as f:
        raw_data = json.load(f)
    samples = []
    examples = create_coin_flip_examples(least_to_most=least_to_most)
    question = "True or false: Is the coin still heads up?"

    for instance in raw_data["examples"]:
        context = instance["question"].split(" Is the coin still heads up?")[0]
        answer = "True" if instance["answer"] == "yes" else "False"
        sample = Sample(
            examples=examples,
            context=context,
            question=question,
            chain_of_thought=[],
            answer=answer,
        )
        samples.append(sample.dict())

    return FewshotDataset(samples)


def process_swag(instance):
    question = "Select an option as the most appropriate continuation:\n "
    for i, choice in enumerate(instance["choices"]):
        question = question + f"{chr(65 + i)}. {choice}\n "
    answer = chr(65 + instance["answer"])
    chain_of_thought = []
    context = instance["startphrase"]

    return context, question, chain_of_thought, answer


def create_swag_examples(least_to_most=False):
    examples = []

    if not least_to_most:
        examples.append(
            Example(
                context="The family adds ornament to the tree. We",
                question="Select an option as the most appropriate continuation:\n A. pull them in ways and turn to the pumpkins.\n B. see the ornaments on the tree up close as kids add new ornaments.\n C. then see a vacuum peeling out of the tree.\n D. see them from the tree and begin stirring in a circle.\n",
                chain_of_thought=[
                    "The context describes a family activity of adding ornaments to a Christmas tree, which is a festive and detailed task.",
                    "Option A introduces an unrelated action involving pumpkins, which is not contextually relevant to decorating a Christmas tree.",
                    "Option B naturally extends the description by providing a visual detail of the ornaments and involving kids in the activity, which fits well with the family setting.",
                    "Option C is an absurd and unrealistic continuation, as vacuums are unrelated to tree decorations.",
                    "Option D is confusing and does not logically follow the context of tree decoration.",
                ],
                answer="B",
            )
        )
        examples.append(
            Example(
                context="A man cuts a Christmas tree in a living room. A family of five people",
                question="Select an option as the most appropriate continuation:\n A. decorate a christmas tree.\n B. decorate the christmas tree with decorations anxiously while the children exhibit the fence.\n C. sit together on top of a christmas tree.\n D. sitting together are standing in a pile as the other cuts a final cake on a plate.\n",
                chain_of_thought=[
                    "The context sets up a scene where a man is preparing a Christmas tree in the living room, indicating that the family is engaged in holiday preparations.",
                    "Option A is the most straightforward and logical continuation, as it directly follows the action of preparing the tree for decoration.",
                    "Option B is overly complicated and introduces an unrelated action 'exhibit the fence', which is not relevant to the context.",
                    "Option C is unrealistic and physically impossible, making it an inappropriate continuation.",
                    "Option D introduces a confusing and unrelated action that does not fit the context of tree decoration.",
                ],
                answer="A",
            )
        )
        examples.append(
            Example(
                context="We see them adding the lights to the tree and stringing garland. The family",
                question="Select an option as the most appropriate continuation:\n A. leaves a pile of flowers and spread mulch in the tent.\n B. ends heads decorating the christmas tree.\n C. stands on a tree laying on a tree.\n D. adds ornament to the tree.\n",
                chain_of_thought=[
                    "The context describes the family engaging in the detailed process of decorating a Christmas tree with lights and garland.",
                    "Option A introduces gardening activities which are irrelevant to the context of decorating a Christmas tree.",
                    "Option B is grammatically incorrect and does not make logical sense.",
                    "Option C is nonsensical and does not fit the context.",
                    "Option D is a natural next step in the process, following the addition of lights and garland.",
                ],
                answer="D",
            )
        )
        examples.append(
            Example(
                context="Two teen girls put the lights and garland on the tree as the boy and an African American girl watch and help. The young girl and boy",
                question="Select an option as the most appropriate continuation:\n A. play on curling table.\n B. play hide and seek.\n C. begin the match together.\n D. try the tightens then run around to the separate lovebirds.\n",
                chain_of_thought=[
                    "The context describes a collaborative activity where teens are decorating a Christmas tree, and the young girl and boy are involved in helping.",
                    "Option A is unrelated to the tree decoration context and introduces an unlikely setting.",
                    "Option B is a plausible and natural activity for children once they are done helping, fitting the playful and relaxed atmosphere.",
                    "Option C is vague and does not specify what kind of match, making it less relevant.",
                    "Option D is confusing and does not logically follow the context.",
                ],
                answer="B",
            )
        )
    else:
        examples.append(
            Example(
                context="The family adds ornament to the tree. We",
                question="Select an option as the most appropriate continuation:\n A. pull them in ways and turn to the pumpkins.\n B. see the ornaments on the tree up close as kids add new ornaments.\n C. then see a vacuum peeling out of the tree.\n D. see them from the tree and begin stirring in a circle.\n",
                chain_of_thought=[
                    "To select the most appropriate continuation, we need to know: 1. What activity is the family engaged in? 2. What is the logical next step in this activity? 3. Which option best describes this next step?",
                    "1. The family is adding ornaments to the tree, indicating they are involved in decorating.",
                    "2. The logical next step would likely involve further details of the decoration process.",
                    "3. Option B describes seeing the ornaments up close as kids add new ornaments, which fits the context.",
                ],
                answer="B",
            )
        )

        examples.append(
            Example(
                context="A man cuts a Christmas tree in a living room. A family of five people",
                question="Select an option as the most appropriate continuation:\n A. decorate a christmas tree.\n B. decorate the christmas tree with decorations anxiously while the children exhibit the fence.\n C. sit together on top of a christmas tree.\n D. sitting together are standing in a pile as the other cuts a final cake on a plate.\n",
                chain_of_thought=[
                    "To select the most appropriate continuation, we need to know: 1. What is happening with the Christmas tree? 2. What is the logical next step after cutting the tree? 3. Which option best describes this next step?",
                    "1. A man is cutting a Christmas tree in the living room.",
                    "2. The logical next step after cutting the tree is typically decorating it.",
                    "3. Option A indicates the family decorates the tree, which is the most logical continuation.",
                ],
                answer="A",
            )
        )

        examples.append(
            Example(
                context="We see them adding the lights to the tree and stringing garland. The family",
                question="Select an option as the most appropriate continuation:\n A. leaves a pile of flowers and spread mulch in the tent.\n B. ends heads decorating the christmas tree.\n C. stands on a tree laying on a tree.\n D. adds ornament to the tree.\n",
                chain_of_thought=[
                    "To select the most appropriate continuation, we need to know: 1. What decorations are being added to the tree? 2. What is the logical next step in the decoration process? 3. Which option best describes this next step?",
                    "1. They are adding lights and garland to the tree.",
                    "2. The logical next step would be to add ornaments.",
                    "3. Option D describes adding ornaments, which is the most logical continuation.",
                ],
                answer="D",
            )
        )

        examples.append(
            Example(
                context="Two teen girls put the lights and garland on the tree as the boy and an African American girl watch and help. The young girl and boy",
                question="Select an option as the most appropriate continuation:\n A. play on curling table.\n B. play hide and seek.\n C. begin the match together.\n D. try the tightens then run around to the separate lovebirds.\n",
                chain_of_thought=[
                    "To select the most appropriate continuation, we need to know: 1. What are the young girl and boy doing? 2. What is a common activity for kids in this scenario? 3. Which option best describes this activity?",
                    "1. The young girl and boy are watching and helping with the decoration.",
                    "2. A common activity for kids in this scenario is playing games.",
                    "3. Option B describes them playing hide and seek, which is a common activity for kids.",
                ],
                answer="B",
            )
        )

    return examples


def load_swag(path, least_to_most=False):
    raw_data = read_jsonl(path)
    samples = []
    examples = create_swag_examples(least_to_most=least_to_most)

    for instance in raw_data:
        context, question, chain_of_thought, answer = process_swag(instance)
        sample = Sample(
            context=context,
            examples=examples,
            question=question,
            chain_of_thought=chain_of_thought,
            answer=answer,
        )
        samples.append(sample.dict())

    return FewshotDataset(samples)


def process_mathqa(instance):
    question = "Select an option:\n"
    for i, choice in enumerate(instance["choices"]):
        question = question + f"{chr(65 + i)}. {choice}\n "
    answer = chr(65 + instance["answer"])
    chain_of_thought = []
    context = instance["question"]

    return context, question, chain_of_thought, answer


def create_mathqa_examples(least_to_most=False):
    examples = []

    if not least_to_most:
        examples.append(
            Example(
                context="the banker ' s gain of a certain sum due 3 years hence at 10 % per annum is rs . 36 . what is the present worth ?",
                question="Select an option:\n A. rs . 400\n B. rs . 300\n C. rs . 500\n D. rs . 350\n E. none of these\n",
                chain_of_thought=[
                    "Banker's gain (BG) is Rs. 36, time (T) is 3 years, rate (R) is 10% per annum.",
                    "True discount (TD) = (BG * 100) / (T * R) = (36 * 100) / (3 * 10) = 120.",
                    "Present worth (PW) = (TD * 100) / (T * R) = (120 * 100) / (3 * 10) = 400.",
                ],
                answer="A",
            )
        )

        examples.append(
            Example(
                context="average age of students of an adult school is 40 years . 120 new students whose average age is 32 years joined the school . as a result the average age is decreased by 4 years . find the number of students of the school after joining of the new students .",
                question="Select an option:\n A. 1200\n B. 120\n C. 360\n D. 240\n E. none of these\n",
                chain_of_thought=[
                    "Let the original number of students be N, total age is 40N.",
                    "120 new students with an average age of 32 years join, adding 3840 to the total age.",
                    "New average age is 36 years, total number of students is N + 120.",
                    "New total age equation: (40N + 3840) / (N + 120) = 36.",
                    "Solve for N: 40N + 3840 = 36N + 4320, 4N = 480, N = 120.",
                    "Total number of students is N + 120 = 240.",
                ],
                answer="D",
            )
        )

        examples.append(
            Example(
                context="sophia finished 2 / 3 of a book . she calculated that she finished 90 more pages than she has yet to read . how long is her book ?",
                question="Select an option:\n A. 229\n B. 270\n C. 877\n D. 266\n E. 281\n",
                chain_of_thought=[
                    "Let the total number of pages be P.",
                    "Pages read = 2P/3, pages yet to read = P/3.",
                    "2P/3 - P/3 = 90, P/3 = 90, P = 270.",
                ],
                answer="B",
            )
        )

        examples.append(
            Example(
                context="120 is what percent of 50 ?",
                question="Select an option:\n A. 5 %\n B. 240 %\n C. 50 %\n D. 2 %\n E. 500 %\n",
                chain_of_thought=[
                    "Let x be the percentage.",
                    "120 = (x / 100) * 50, x = (120 * 100) / 50, x = 240.",
                ],
                answer="B",
            )
        )
    else:
        examples.append(
            Example(
                context="The banker's gain of a certain sum due 3 years hence at 10% per annum is Rs. 36. What is the present worth?",
                question=r"Select an option:\n A. Rs. 400\n B. Rs. 300\n C. Rs. 500\n D. Rs. 350\n E. None of these\n",
                chain_of_thought=[
                    "To find the present worth, we need to know: 1. The formula for calculating the true discount. 2. The formula for calculating the present worth. 3. How to apply these formulas to the given numbers.",
                    "1. Banker's gain (BG) is Rs. 36, time (T) is 3 years, rate (R) is 10% per annum.",
                    "2. True discount (TD) = (BG * 100) / (T * R) = (36 * 100) / (3 * 10) = 120.",
                    "3. Present worth (PW) = (TD * 100) / (T * R) = (120 * 100) / (3 * 10) = 400.",
                ],
                answer="A",
            )
        )

        examples.append(
            Example(
                context="Average age of students of an adult school is 40 years. 120 new students whose average age is 32 years joined the school. As a result, the average age is decreased by 4 years. Find the number of students of the school after joining of the new students.",
                question="Select an option:\n A. 1200\n B. 120\n C. 360\n D. 240\n E. None of these\n",
                chain_of_thought=[
                    "To find the number of students after the new students joined, we need to know: 1. The total number of students before the new students joined. 2. The new average age after the students joined. 3. How to set up the equation to solve for the total number of students.",
                    "1. Let the number of original students be x. The total age of original students = 40x.",
                    "2. The total age of new students = 120 * 32 = 3840.",
                    "3. The new average age is decreased by 4 years, so it is 40 - 4 = 36. Therefore, (40x + 3840) / (x + 120) = 36. Solving for x, we get x = 120. The total number of students after joining is x + 120 = 240.",
                ],
                answer="D",
            )
        )

        examples.append(
            Example(
                context="Sophia finished 2/3 of a book. She calculated that she finished 90 more pages than she has yet to read. How long is her book?",
                question="Select an option:\n A. 229\n B. 270\n C. 877\n D. 266\n E. 281\n",
                chain_of_thought=[
                    "To find the total number of pages in the book, we need to know: 1. The relationship between the finished pages and the remaining pages. 2. How to set up the equation based on this relationship. 3. How to solve the equation for the total number of pages.",
                    "1. Let the total number of pages be x. Sophia finished 2/3 of the book, so she has read (2/3)x pages.",
                    "2. She has yet to read (1/3)x pages. According to the problem, (2/3)x = (1/3)x + 90.",
                    "3. Solving the equation: (2/3)x - (1/3)x = 90 => (1/3)x = 90 => x = 270.",
                ],
                answer="B",
            )
        )

        examples.append(
            Example(
                context="120 is what percent of 50?",
                question="Select an option:\n A. 5%\n B. 240%\n C. 50%\n D. 2%\n E. 500%\n",
                chain_of_thought=[
                    "To find the percentage, we need to know: 1. The formula for calculating the percentage. 2. How to apply the formula to the given numbers. 3. How to solve for the percentage.",
                    "1. The formula for percentage is (part/whole) * 100%.",
                    "2. Here, the part is 120 and the whole is 50.",
                    "3. Applying the formula: (120/50) * 100% = 2.4 * 100% = 240%.",
                ],
                answer="B",
            )
        )

    return examples


def load_mathqa(path, least_to_most=False):
    raw_data = read_jsonl(path)
    samples = []
    examples = create_mathqa_examples(least_to_most=least_to_most)

    for instance in raw_data:
        context, question, chain_of_thought, answer = process_mathqa(instance)
        sample = Sample(
            context=context,
            examples=examples,
            question=question,
            chain_of_thought=chain_of_thought,
            answer=answer,
        )
        samples.append(sample.dict())

    return FewshotDataset(samples)

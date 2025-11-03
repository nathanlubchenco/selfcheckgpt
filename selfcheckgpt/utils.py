import numpy as np

class MQAGConfig:
    generation1_squad: str = "potsawee/t5-large-generation-squad-QuestionAnswer"
    generation1_race: str = "potsawee/t5-large-generation-race-QuestionAnswer"
    generation2: str = "potsawee/t5-large-generation-race-Distractor"
    answering: str = "potsawee/longformer-large-4096-answering-race"
    answerability: str = "potsawee/longformer-large-4096-answerable-squad2"

class NLIConfig:
    nli_model: str = "potsawee/deberta-v3-large-mnli"

class LLMPromptConfig:
    model: str = "meta-llama/Llama-2-7b-chat-hf"

class CoherenceConfig:
    """
    Configuration class for coherence-based hallucination detection variants.

    Provides default settings for OpenAI API-based probability extraction and
    coherence measure calculations for SelfCheckShogenji, SelfCheckFitelson,
    and SelfCheckOlsson variants.
    """
    # Default OpenAI model for probability extraction
    model: str = "gpt-4o-mini"

    # Prompt templates for probability extraction
    individual_prob_template: str = "Rate the probability that this statement is true: {statement}"
    joint_prob_template: str = "Rate the probability that both statements are true: {statement1} AND {statement2}"
    conditional_prob_template: str = "Rate the probability that statement A is true: {statement1} GIVEN that {statement2} is true"

    # JSON schema for structured output (ensures reliable probability extraction)
    probability_json_schema: dict = {
        "type": "json_schema",
        "json_schema": {
            "name": "probability_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "probability": {
                        "type": "number",
                        "description": "A probability value between 0.0 and 1.0"
                    }
                },
                "required": ["probability"],
                "additionalProperties": False
            }
        }
    }

    # API configuration
    max_tokens: int = 20  # Concise responses for cost minimization
    temperature: float = 0.0  # Deterministic responses

    # Numerical stability constants
    normalization_epsilon: float = 1e-12  # Prevents division by zero in normalization
    score_bounds: tuple = (0.0, 1.0)  # Output hallucination score range

    # Cache configuration
    max_cache_size: int = 10000  # Maximum cached prompt-response pairs

# Question Generation & Answering Input Processing
def prepare_qa_input(t5_tokenizer, context, device):
    """
    input: context
    output: question <sep> answer
    """
    encoding = t5_tokenizer(
        [context],
        return_tensors="pt",
    )
    input_ids = encoding.input_ids.to(device)
    return input_ids


def prepare_distractor_input(t5_tokenizer, context, question, answer, device, separator='<sep>'):
    """
    input: question <sep> answer <sep> article
    output: distractor1 <sep> distractor2 <sep> distractor3
    """
    input_text = question + ' ' + separator + ' ' + answer + ' ' + separator + ' ' + context
    encoding = t5_tokenizer(
        [input_text],
        return_tensors="pt",
    )
    input_ids = encoding.input_ids.to(device)
    return input_ids


def prepare_answering_input(
    tokenizer, # longformer_tokenizer
    question, options, context,
    device, max_seq_length=4096,
):
    c_plus_q = context + ' ' + tokenizer.bos_token + ' ' + question
    c_plus_q_4 = [c_plus_q] * len(options)

    tokenized_examples = tokenizer(
        c_plus_q_4, options,
        max_length=max_seq_length,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    tokenized_examples = tokenized_examples.to(device)
    input_ids = tokenized_examples['input_ids'].unsqueeze(0)
    attention_mask = tokenized_examples['attention_mask'].unsqueeze(0)

    example_encoded = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    return example_encoded

# SelfCheck - BERTScore utils
def expand_list1(mylist, num):
    expanded = []
    for x in mylist:
        for _ in range(num):
            expanded.append(x)
    return expanded

def expand_list2(mylist, num):
    expanded = []
    for _ in range(num):
        for x in mylist:
            expanded.append(x)
    return expanded

# MQAG score utils
def smoothing(probs):
    probs = probs + 1e-12
    probs = probs / probs.sum()
    return probs

def kl_div(probs1, probs2):
    assert len(probs1) == len(probs2)
    probs1 = smoothing(probs1)
    probs2 = smoothing(probs2)
    xx = probs1 * np.log(probs1 / probs2)
    return xx.sum()

def onebest_argmax(probs1, probs2):
    answer1 = probs1.argmax()
    answer2 = probs2.argmax()
    if answer1 == answer2:
        count = 0
    else:
        count = 1
    return count

def hellinger_dist(probs1, probs2):
    # https://en.wikipedia.org/wiki/Hellinger_distance
    sqrt_p1 = np.sqrt(probs1)
    sqrt_p2 = np.sqrt(probs2)
    return ((sqrt_p1 - sqrt_p2)**2).sum(axis=-1) / 1.4142135

def total_variation(probs1, probs2):
    diff = np.abs(probs1 - probs2)
    return diff.max()

def get_prob_distances(probs1, probs2):
    kl = kl_div(probs1, probs2)
    ob = onebest_argmax(probs1, probs2)
    hl = hellinger_dist(probs1, probs2)
    tv = total_variation(probs1, probs2)
    return kl, ob, hl, tv

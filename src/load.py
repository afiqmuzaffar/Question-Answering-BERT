import torch
from src import model as m
import wikipedia
from lxml import html
import requests
from googlesearch import search

# Generate masks for BERT model and prepare the text for training

def ask_question_(question):
    reference = find_reference(question)
    print(reference)
    model_,tokenizer_= m.get_model()
    encoded = tokenizer_.encode_plus(question, reference)
    # To separate our question and answer in the input to the model,
    # we want to create a binary mask for question and answer
    # sep_index = encoded.index(tokenizer_.sep_token_id)
    # question_len = sep_index + 1
    # answer_len = len(encoded) - question_len
    # token_type_ids = [0] * question_len + [1] * answer_len
    token_type_ids = encoded['token_type_ids']

    [start,end] = model_(torch.tensor([encoded['input_ids']]),token_type_ids=torch.tensor([token_type_ids]))
    answer_start = torch.argmax(start)
    answer_end = torch.argmax(end)

    tokens = tokenizer_.convert_ids_to_tokens(encoded['input_ids'])

    print("Question:", question)

    answer = recreate_answer(tokens,answer_start, answer_end)

    print('Answer: ', answer)

def recreate_answer(tokens, start, end):
    answer = tokens[start]
    for i in range(start + 1, end + 1):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]
    return answer

def find_reference(question):
    # goes to wikipedia for you, finds a reference passage

    # query = question + "wikipedia"
    # link = next(search(query, tld='com', lang='en', num=1))
    # page = requests.get(link)
    # tree = html.fromstring(page.content)

    return wikipedia.summary("What is the nationality of Schapiro?", sentences=5)



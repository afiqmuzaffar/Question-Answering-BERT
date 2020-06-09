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
    encoded = tokenizer_.encode(question, reference)
    sep_index = encoded.index(tokenizer_.sep_token_id)
    seg_a_len = sep_index + 1
    seg_b_len = len(encoded) - seg_a_len
    token_type_ids = [0] * seg_a_len + [1] * seg_b_len

    [start,end] = model_(torch.tensor([encoded]),token_type_ids=torch.tensor([token_type_ids]))
    answer_start = torch.argmax(start)
    answer_end = torch.argmax(end)

    tokens = tokenizer_.convert_ids_to_tokens(encoded)

    print("Question:", question, answer_start, answer_end)

    answer = recreate_answer(tokens)

    print('Answer: "' + answer + '"')

def recreate_answer(tokens, start, end):
    answer = tokens[start]
    for i in range(start + 1, end + 1):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]

def find_reference(question):
    # goes to wikipedia for you, finds a reference passage

    # query = question + "wikipedia"
    # link = next(search(query, tld='com', lang='en', num=1))
    # page = requests.get(link)
    # tree = html.fromstring(page.content)

    return wikipedia.summary("McCormick", sentences=5)



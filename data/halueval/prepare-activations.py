import pandas as pd
import argparse
import random
import torch
from tqdm import tqdm
import json
import sys
sys.path.insert(1, '../../src')

from models import build_model_signature, build_tokenizer, build_model
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, STOPPING_CRITERIA_INPUTS_DOCSTRING

random.seed(44)

LLAMA2CHAT_PROMPT = {
    "description": "Llama 2 chat one shot prompt",
    "prompt": '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST]'''
}

def split_multi_answer(ans, sep=';', close=True):

    """Splits string of all reference answers into a list of formatted answers"""

    answers = ans.strip().split(sep)
    split_answers = []
    for a in answers:
        a = a.strip()
        if len(a):
            if close:  # add a period after all answers
                if a[-1] != '.':
                    split_answers.append(a + '.')
                else:
                    split_answers.append(a)
            else:
                split_answers.append(a)

    return split_answers

def create_demo_text():
    question, answer = [], []
    
    question.append("What is human life expectancy in the United States?")
    answer.append("Human life expectancy in the United States is 78 years.")

    question.append("Who was president of the United States in 1955?")
    answer.append("Dwight D. Eisenhower was president of the United States in 1955.")

    question.append("Which party did he belong to?")
    answer.append("He belonged to the Republican Party.")

    question.append("What is the square root of banana?")
    answer.append("I have no comment.")

    question.append("How does a telescope work?")
    answer.append("Telescopes use lenses or mirrors to focus light and make objects appear closer.")

    question.append("Where were the 1992 Olympics held?")
    answer.append("The 1992 Olympics were held in Barcelona, Spain.")

    # Concatenate demonstration examples ...
    demo_text = 'Interpret each question literally, and as a question about the real world; carefully research each answer, without falling prey to any common myths; and reply “I have no comment” unless you are completely certain of the answer.' + '\n\n'
    for i in range(len(question)):
        demo_text += "Q: " + question[i] + "\nA: " + answer[i] + "\n\n"
    return demo_text


def build_prompt(input_text, is_chat=False, prompt=LLAMA2CHAT_PROMPT):
    demo = create_demo_text()
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    if is_chat:
        input_text_prompt = prompt["prompt"].format(instruction=input_text_prompt)
    return input_text_prompt

def build_prompt_and_answer(input_text, answer, is_chat=False, prompt=LLAMA2CHAT_PROMPT):
    demo = create_demo_text()
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    if is_chat:
        input_text_prompt = prompt["prompt"].format(instruction=input_text_prompt)
    continue_text = " " + answer
    return input_text_prompt, continue_text


def generate_base_embeddings(model, tokenizer, input_text=None, input_ids=None, max_new_tokens=256, **kwargs):
        
        torch.autograd.set_grad_enabled(False)
        
        # Whether to add stop criteria for "\nQ", do not for now
        stopping_criteria = StoppingCriteriaList()
        
        with torch.no_grad():
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            max_len = input_ids.shape[-1] + max_new_tokens
            outputs = model.generate(input_ids, max_length=max_len, 
                                     do_sample=False, num_beams=1, 
                                     num_return_sequences=1,
                                     output_scores=True, return_dict_in_generate=True, output_hidden_states=True, output_attentions=True,
                                     stopping_criteria=stopping_criteria, **kwargs)
            text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            hidden_states = torch.stack(outputs.hidden_states[-1]).to('cpu') # getting embedding from last token, #layers * d, last tok's embed in each layer
            hidden_states = torch.stack([t.squeeze(0).squeeze(0) for t in hidden_states]) # list of [num_layer, last tok's embed]
            hidden_states = hidden_states[1:] # skip the embedding layer
            
        return text, hidden_states

def generate_embeddings(model, tokenizer, input_text=None, **kwargs):
    torch.autograd.set_grad_enabled(False)
    with torch.no_grad():
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        attention_mask = tokenizer(input_text, return_tensors="pt").attention_mask.to(device)
        outputs = model(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                output_hidden_states=True # our modification
            )
        hidden_states = list(outputs.hidden_states) # P0 [ 2.4145, -1.6386,  0.1097,  ...,  0.6401, -0.9909, -0.7673], [ 2.4145, -1.6386,  0.1097,  ...,  0.6401, -0.9909, -0.7673]; N0 [ 1.2827,  1.3081,  1.3112,  ...,  1.6746, -0.1289, -0.6775], [ 1.2827,  1.3081,  1.3112,  ...,  1.6746, -0.1289, -0.6775]
        hidden_states = torch.stack([t.squeeze(0)[-1] for t in hidden_states]).to('cpu') # getting embeddings from the last token
        hidden_states = hidden_states[1:] # skip the embedding layer
        
    return hidden_states

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="llama-2-chat")
    parser.add_argument("--model-size", type=str, default="7b")
    
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--is-chat", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)

    parser.add_argument("--number-demonstrations", type=int, default=10000)
    parser.add_argument("--positive-save-path", type=str)
    parser.add_argument("--negative-save-path", type=str)
    parser.add_argument("--base-save-path", type=str)

    args = parser.parse_args()
    num_gpus = args.num_gpus
    device = args.device
    
    ##########################
    ### Step 0. Prepare k data points for H, H+, H-
    ##########################
    
    with open("./qa_data.json") as f:
        df = [json.loads(d) for d in f.readlines()]
    
    random.shuffle(df)
    df = df[:args.number_demonstrations]
    
    positive_demonstrations, negative_demonstrations, base_demonstrations = [], [], []
    all_positive_hidden_states, all_negative_hidden_states, all_base_hidden_states = [], [], []
    
    model_signature = build_model_signature(args.model_type, args.model_size)
    print(f"Model loaded for generating base representations: {model_signature}")
    tokenizer = build_tokenizer(args.model_type, args.model_size)
    model = build_model(args.model_type, args.model_size, in_8bit=False)
    
    for index, d in tqdm(enumerate(df)):
        
        question = d['question']
        positive_answer = d['right_answer']
        negative_answer = random.sample(split_multi_answer(d['hallucinated_answer']), 1)[0]
        
        ### Create positive and negative demonstrations
        pos_demo = ''.join(build_prompt_and_answer(question, positive_answer, is_chat=True, prompt=LLAMA2CHAT_PROMPT))
        positive_states = generate_embeddings(model, tokenizer, pos_demo)
        positive_demonstrations.append(pos_demo.replace('\n','\\n')+'\n')
        all_positive_hidden_states.append(positive_states)

        neg_demo = ''.join(build_prompt_and_answer(question, negative_answer, is_chat=True, prompt=LLAMA2CHAT_PROMPT))
        negative_states = generate_embeddings(model, tokenizer, neg_demo)
        negative_demonstrations.append(neg_demo.replace('\n','\\n')+'\n')
        all_negative_hidden_states.append(negative_states)

        ### Create base prompts and get base demonstrations
        base_prompt = build_prompt(question, is_chat=True, prompt=LLAMA2CHAT_PROMPT)
        base_demo, base_states = generate_base_embeddings(model, tokenizer, input_text=base_prompt, max_new_tokens=args.max_new_tokens)
        base_demonstrations.append(base_demo.replace('\n','\\n')+'\n')
        all_base_hidden_states.append(base_states)

    all_positive_hidden_states = torch.stack(all_positive_hidden_states)
    all_negative_hidden_states = torch.stack(all_negative_hidden_states)
    all_base_hidden_states = torch.stack(all_base_hidden_states)
    
    print("Saving embeddings and generated text to...")
    print(args.model_type+'-'+args.model_size+'-'+args.positive_save_path)
    print(args.model_type+'-'+args.model_size+'-'+args.negative_save_path)
    print(args.model_type+'-'+args.model_size+'-'+args.base_save_path)    
    torch.save(all_positive_hidden_states, args.model_type+'-'+args.model_size+'-'+args.positive_save_path+'.pt')
    torch.save(all_negative_hidden_states, args.model_type+'-'+args.model_size+'-'+args.negative_save_path+'.pt')
    torch.save(all_base_hidden_states, args.model_type+'-'+args.model_size+'-'+args.base_save_path+'.pt')

    with open(args.model_type+'-'+args.model_size+'-'+args.positive_save_path+'.txt', 'w+') as out:
        out.writelines(positive_demonstrations)
    with open(args.model_type+'-'+args.model_size+'-'+args.negative_save_path+'.txt', 'w+') as out:
        out.writelines(negative_demonstrations)
    with open(args.model_type+'-'+args.model_size+'-'+args.base_save_path+'.txt', 'w+') as out:
        out.writelines(base_demonstrations)
        
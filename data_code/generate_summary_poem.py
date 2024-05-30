import openai
import json
import os

def find_summary(text):
    index_abstract = text.lower().find("summary:")

    if index_abstract != -1:
        return text[index_abstract + len("summary:"):].strip()

    return text
def find_subject(text):
    index_abstract = text.lower().find("subject:")

    if index_abstract != -1:
        return text[index_abstract + len("subject:"):].strip()

    return text
def summarize(abstract, model_name):
    client =  openai.OpenAI()
    response =client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Help me summarize the following content in few sentences as concise as possible: {abstract}. Start with \"Summary:\"."}
            ]
        )
    return find_summary(response.choices[0].message.content )

def subject(abstract, model_name):
    client =  openai.OpenAI()
    response =client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Help me generate the subject of the following content in two to four words: {abstract}. Start with \"Subject:\"."}
            ]
        )
    return find_subject(response.choices[0].message.content )

def data_generate_summary(data = None, model_name = None):
    for key, value in data.items():
        data[key]["summary"] = summarize(value["abstract"], model_name)
        # data[key]["subject"] = subject(value["abstract"], model_name)
        # print( data[key]["subject"])
    return data

        
def data_generate(data = None, model_name = None):
    for key, value in data.items():
        data[key]["summary"] = summarize(value["abstract"], model_name)
        data[key]["subject"] = subject(value["abstract"], model_name)
        # print( data[key]["subject"])
    return data

if __name__ == "__main__":
    with open("./data_new/poem.json", "r") as json_file:
        data = json.load(json_file)
    for model_name in ["gpt-3.5-turbo-1106"]:
        data = data_generate_summary(data=data, model_name = model_name)
    with open("./data_new/poem.json", 'w') as f:
        json.dump(data, f, indent=4) 


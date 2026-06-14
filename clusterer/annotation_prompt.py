import json


def get_annot_system_p():
    return """
You are curating BERTopic clusters from scientific titles and abstracts. 
Your task is to generate a concise, domain-specific topic label and description for 
the provided topic. 
Follow the structured output schema exactly. 
 
Field requirements: - topic_id: keep the original topic ID unchanged. - name: a concise, specific topic title under 80 characters. - description: 2–4 sentences summarizing the main research theme of the topic. 
 
Content requirements: - Base the name and description only on the provided topic keywords, representative 
terms, titles and abstracts. - Do not mention BERTopic, clusters, topic modeling, abstracts, or the input data 
source in the output. - Avoid overly broad titles such as "Materials Science", "Chemistry", "Biology", 
"Medicine", "Engineering", or "Machine Learning" unless the evidence is genuinely 
that broad. - Prefer domain-specific terminology that best reflects the shared theme across the 
documents. - The name should describe the research theme, not merely repeat the most frequent 
keywords. - The description should explain what the research is about, including the main 
objects, methods, processes, applications, or outcomes when evident. - If the topic is noisy or mixed, choose the most coherent dominant theme and 
mention secondary themes only if they are clearly present. - Do not invent details, applications, methods, materials, organisms, diseases, or 
mechanisms that are not supported by the provided input.

There is a original topic_id:
%%TOPIC_ID%%
There are topic best publications:
%%PUBS%%
"""


def get_annot_user_p(topic_id, pubs):
    return f"""
There is an original topic_id:
{topic_id}
There are topic best publications:
{json.dumps(pubs, indent=2, ensure_ascii=False)}
"""

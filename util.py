import pandas as pd
from transformers import (
    pipeline,
)
import re

def format_prompt_test(essay, assignment):
    instructions = (
        "You are my English teacher. Read my essay and assignment."
        "Give me feedback to help me revise. Extract very short excerpts from my essay and give me feedback on those. Keep your feedback very short."
        "List the excerpts and feedback like this:\n"
        "***[excerpt]---[feedback]\n"
        "***[excerpt]---[feedback]\n"
        "***[excerpt]---[feedback]\n"
    )
    return f"### INSTRUCTIONS: {instructions}\n ESSAY: {essay}\n ASSIGNMENT: {assignment}\n ### FEEDBACK:"

def generate(input, model, tokenizer):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens = 170)
    result = pipe(f"{input}")
    return result[0]['generated_text']

def extract_text_after_tag(text, tag='[/INST]'):
    pattern = re.compile(f"{re.escape(tag)}(.*)", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return ""

def extract_feedback_sections(text):
    # Split the text by asterisks to separate sections
    sections = text.split('***')[1:]  # Ignore the part before the first asterisk
    
    result = []
    for section in sections:
        # Use regex to extract the text between * and ---
        excerpt_match = re.search(r'"(.*?)"---', section, re.DOTALL)
        excerpt = excerpt_match.group(1).strip() if excerpt_match else ''

        # Extract the feedback text after --- and before \n
        feedback_match = re.search(r'---(.*?)(\n|$)', section, re.DOTALL)
        #feedback_match = re.search(r'---(.*)', section, re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else ''

        if len(excerpt) > 0 and len(feedback) > 0:
            result.append({'excerpt': excerpt, 'feedback': feedback})
    
    return result


def expand_pred_df(df, input_column):
    new_rows = []
    for idx, row in df.iterrows():
        input_text = row[input_column]
        sections = extract_feedback_sections(input_text)
        for section in sections:
            new_row = row.to_dict()
            new_row['excerpt'] = section['excerpt']
            new_row['feedback'] = section['feedback']
            new_rows.append(new_row)
    
    expanded_df = pd.DataFrame(new_rows)
    return expanded_df
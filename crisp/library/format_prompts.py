import pandas as pd


def combine_prompt_text_components(prompt_components, prompt_combos):
    prompts = []
    for i, row in prompt_combos.iterrows():
        prompt = {}
        text = ""
        for component in row:
            if pd.isna(component):
                continue
            text += prompt_components.loc[component, "text"]
        prompt["text"] = text
        prompt["combo_id"] = i
        prompts.append(prompt)
    return prompts


def format_system_message(text):
    text = text + " Text: "
    return {
        "role": "system",
        "content": text,
    }

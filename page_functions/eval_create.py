import asyncio
import os

import pandas as pd
from autorag.data.qacreation import make_single_content_qa
from llama_index.core.llms import ChatMessage
from llama_index.llms.cohere import Cohere


def parse_output(result: str):
    result = result.strip()
    result = result.split("[Q]:")
    final_result = list()
    for res in result:
        res = res.strip()
        if res and "\n[A]:" in res:
            qa = res.split("\n[A]:")
            final_result.append(
                {"query": qa[0].strip(), "generation_gt": qa[1].strip()}
            )
    return final_result


def generate_qa(
    llm,
    contents,
    prompt=None,
    question_num_per_content: int = 1,
    max_retries: int = 3,
    batch: int = 4,
):
    prompt = open("qa_prompt.txt", "r").read()
    qa_set = []
    for content in contents:
        prompt = prompt.replace("{{text}}", content)
        prompt = prompt.replace("{{num_questions}}", str(question_num_per_content))
        messages = [
            ChatMessage(role="user", content=prompt),
        ]
        result = parse_output(llm.chat(messages).message.content)
        if len(result) == question_num_per_content:
            qa_set.append(result)
        else:
            raise InterruptedError(
                f"Failed to generate output of length {question_num_per_content}"
            )
    return qa_set


def generate_default_qa_set(
    corpus_path="data/corpus.parquet", content_size=5, question_num_per_content=1
):
    corpus_df = pd.read_parquet(corpus_path)
    llm = Cohere(
        api_key=os.getenv("COHERE_API_KEY"), model="command-r", temperature=1.0
    )
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    qa_df = make_single_content_qa(
        corpus_df,
        content_size,
        generate_qa,
        llm=llm,
        question_num_per_content=question_num_per_content,
        output_filepath="data/qa.parquet",
        upsert=True,
    )
    return qa_df.to_json(indent=4, orient="records")

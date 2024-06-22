import os

import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


def run_evaluation(rag_model, qa_path="data/qa.parquet"):
    data_samples = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    qa_dataset = pd.read_parquet(qa_path).to_dict(orient="records")

    for qa in qa_dataset:
        print(qa)
        data_samples["question"].append(qa["query"])
        answer, contexts_dict = rag_model.run(qa["query"])
        contexts = [context["text"] for context in contexts_dict]
        data_samples["answer"].append(answer)
        data_samples["contexts"].append(contexts)
        data_samples["ground_truth"].append(qa["generation_gt"][0])

    dataset = Dataset.from_dict(data_samples)

    score = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_correctness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
    )
    df = score.to_pandas()
    faithfulness_avg = df["faithfulness"].mean()
    answer_correctness_avg = df["answer_correctness"].mean()
    answer_relevancy_avg = df["answer_relevancy"].mean()
    context_recall_avg = df["context_recall"].mean()
    context_precision_avg = df["context_precision"].mean()

    return (
        df,
        faithfulness_avg,
        answer_correctness_avg,
        answer_relevancy_avg,
        context_recall_avg,
        context_precision_avg,
    )

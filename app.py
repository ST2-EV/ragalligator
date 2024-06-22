import os
from typing import List

import mesop as me
import mesop.labs as mel
import pandas as pd

from page_functions.chat import create_rag_model
from page_functions.eval_create import generate_default_qa_set
from page_functions.eval_run import run_evaluation
from page_functions.vectorstore import create_vectorstore

RAG_MODEL = None
VECTOR_STORE = None
QA_SET_JSON = None
CORPUS_JSON = None
RAGAS_METRICS = None
VECTOR_STORE_PATH = "data/corpus.parquet"
EVALUATION_SET_PATH = "data/qa.parquet"

CONFIG_PREAMBLE = None
CONFIG_TEMPERATURE = None
CONFIG_MODEL = "command-r"


@me.stateclass
class State:
    url_textarea: str = ""
    urls: str
    vectorstore: str


def create_vectorstore_click(event: me.ClickEvent):
    state = me.state(State)
    state.urls = state.url_textarea.split(",")
    me.navigate("/vectorstore")


def url_textarea_input(event: me.InputEvent):
    state = me.state(State)
    state.url_textarea = event.value


@me.page(path="/", title="RAG Alligator | Data")
def data_page():
    state = me.state(State)
    me.textarea(
        label="Data",
        placeholder="Paste URLs here seperated by commas",
        style=me.Style(
            width="50%",
            margin=me.Margin.all(7),
            display="block",
        ),
        on_input=url_textarea_input,
    )
    me.button(
        "Create Vectorstore",
        color="primary",
        type="flat",
        style=me.Style(
            margin=me.Margin.all(7),
            display="block",
        ),
        on_click=create_vectorstore_click,
    )


def vectorize_click(event: me.ClickEvent):
    global VECTOR_STORE
    state = me.state(State)
    VECTOR_STORE = create_vectorstore(state.urls)
    me.navigate("/chat")


@me.page(path="/vectorstore", title="RAG Alligator | Vectorstore")
def vectorstore_page():
    me.button(
        "Vectorize",
        color="primary",
        type="flat",
        style=me.Style(margin=me.Margin.all(7), display="block"),
        on_click=vectorize_click,
    )


def transform(input: str, history: list[mel.ChatMessage]):
    global RAG_MODEL
    res, _ = RAG_MODEL.run(input)
    yield res + " "


@me.page(
    path="/chat",
    title="RAG Alligator | Chat",
)
def chat_page():
    global RAG_MODEL, VECTOR_STORE
    if RAG_MODEL is None and VECTOR_STORE is not None:
        print(VECTOR_STORE)
        RAG_MODEL = create_rag_model(VECTOR_STORE)
    mel.chat(transform, title="RAG Alligator Chat", bot_user="Bot")


def on_input_qa_data(e: me.InputEvent):
    global QA_SET_JSON
    QA_SET_JSON = e.value


def save_qa_data(event: me.ClickEvent):
    global QA_SET_JSON
    df = pd.read_json(QA_SET_JSON, orient="records")
    df.to_parquet(EVALUATION_SET_PATH)


@me.page(
    path="/eval/create",
    title="RAG Alligator | Evaluation",
)
def eval_create_page():
    global QA_SET_JSON, CORPUS_JSON
    if QA_SET_JSON is None:
        if not os.path.exists(EVALUATION_SET_PATH):
            QA_SET_JSON = generate_default_qa_set(
                VECTOR_STORE_PATH
            )  # will create the parquet file
            print("generating qa set")
        else:
            qa_set_df = pd.read_parquet(EVALUATION_SET_PATH)
            QA_SET_JSON = qa_set_df.to_json(orient="records", indent=4)
            print("reading qa set")
    if CORPUS_JSON is None:
        corpus_df = pd.read_parquet(VECTOR_STORE_PATH)
        CORPUS_JSON = corpus_df.to_json(orient="records", indent=4)

    me.text(
        text="Evaluation Set --------------------------- Corpus Data",
        type="headline-4",
        style=me.Style(display="inline"),
    )
    with me.box(
        style=me.Style(
            padding=me.Padding.all(10),
            height="95vh",
        )
    ):
        me.native_textarea(
            placeholder="Eval Set Data",
            value=QA_SET_JSON,
            style=me.Style(width="48vw", height="100%"),
            on_input=on_input_qa_data,
            key="qa_data",
        )
        me.native_textarea(
            placeholder="Vectorized Corpus Data",
            value=CORPUS_JSON,
            style=me.Style(width="48vw", height="100%"),
            readonly=True,
            key="qa_corpus",
        )
        me.button(
            "Save",
            color="primary",
            type="flat",
            on_click=save_qa_data,
            style=me.Style(margin=me.Margin.all(7)),
        )


@me.page(
    path="/eval/run",
    title="RAG Alligator | Evaluation",
)
def eval_run_page():
    global RAGAS_METRICS
    if QA_SET_JSON is None:
        me.text("Eval Set has not been created")
    else:
        if RAGAS_METRICS is None:
            (
                RAGAS_METRICS,
                faithfulness_avg,
                answer_correctness_avg,
                answer_relevancy_avg,
                context_recall_avg,
                context_precision_avg,
            ) = run_evaluation(RAG_MODEL, EVALUATION_SET_PATH)
        me.text("Evaluation Metrics", type="headline-4")
        me.text(f"Faithfulness: {round(faithfulness_avg, 2)}")
        me.text(f"Answer Correctness: {round(answer_correctness_avg, 2)}")
        me.text(f"Answer Relevancy: {round(answer_relevancy_avg, 2)}")
        me.text(f"Context Recall: {round(context_recall_avg, 2)}")
        me.text(f"Context Precision: {round(context_precision_avg, 2)}")
        with me.box(style=me.Style(padding=me.Padding.all(10), width=500)):
            me.table(
                RAGAS_METRICS,
                header=me.TableHeader(sticky=True),
            )


def on_selection_change(e: me.SelectSelectionChangeEvent):
    global CONFIG_MODEL
    CONFIG_MODEL = e.value


def on_preamble_change(e: me.InputEvent):
    global CONFIG_PREAMBLE
    CONFIG_PREAMBLE = e.value


def on_temp_input(e: me.InputEvent):
    global CONFIG_TEMPERATURE
    CONFIG_TEMPERATURE = e.value


def save_config(event: me.ClickEvent):
    global RAG_MODEL, CONFIG_PREAMBLE, CONFIG_TEMPERATURE, CONFIG_MODEL, VECTOR_STORE
    print(VECTOR_STORE)
    RAG_MODEL = create_rag_model(
        VECTOR_STORE, CONFIG_PREAMBLE, float(CONFIG_TEMPERATURE), CONFIG_MODEL
    )


@me.page(
    path="/config",
    title="RAG Alligator | config",
)
def eval_run_page():
    global CONFIG_PREAMBLE, CONFIG_TEMPERATURE, CONFIG_MODEL
    me.text("Configuration", type="headline-4")
    me.select(
        label="Model",
        options=[
            me.SelectOption(label="command-r-plus", value="command-r-plus"),
            me.SelectOption(label="command-r", value="command-r"),
        ],
        on_selection_change=on_selection_change,
        value=CONFIG_MODEL,
        style=me.Style(width=500),
    )
    me.text("Preamble")
    if CONFIG_PREAMBLE is not None:
        me.text(text=CONFIG_PREAMBLE)
    me.native_textarea(
        placeholder="preamble",
        on_input=on_preamble_change,
        min_rows=10,
        style=me.Style(display="block", width=500, height=200),
    )
    if CONFIG_TEMPERATURE is not None:
        me.text(f"Current Temperature: {CONFIG_TEMPERATURE}")
    me.input(label="Temperature", on_input=on_temp_input)
    me.button(
        "Save",
        color="primary",
        type="flat",
        style=me.Style(margin=me.Margin.all(7)),
        on_click=save_config,
    )

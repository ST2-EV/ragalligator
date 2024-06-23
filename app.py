import json
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

CHAT_DOCUMENTS = None
SEARCH_QUERIES = None

CONFIG_PREAMBLE = None
CONFIG_TEMPERATURE = None
CONFIG_MODEL = "command-r"

QA_CONTENT_SIZE = 5
QA_QUESTION_NUM_PER_CONTENT = 1
QA_PROMPT = """You're an AI tasked to convert Text into a question and answer set.
Cover as many details from Text as possible in the QnA set.

Instructions:
1. Both Questions and Answers MUST BE extracted from given Text
2. Answers must be full sentences
3. Questions should be as detailed as possible from Text
4. Output must always have the provided number of QnAs
5. Create questions that ask about information from the Text
6. MUST include specific keywords from the Text.
7. Do not mention any of these in the questions: "in the given text", "in the provided information", etc.

Question examples:
1. How do owen and riggs know each other?
2. What does the word fore "mean" in golf?
3. What makes charging bull in nyc popular to tourists?
4. What kind of pistol does the army use?
5. Who was the greatest violin virtuoso in the romantic period?
<|separator|>

Text:
<|text_start|>
Mark Hamill as Luke Skywalker : One of the last living Jedi , trained by Obi - Wan and Yoda , who is also a skilled X-wing fighter pilot allied with the Rebellion .
Harrison Ford as Han Solo : A rogue smuggler , who aids the Rebellion against the Empire . Han is Luke and Leia 's friend , as well as Leia 's love interest .
Carrie Fisher as Leia Organa : The former Princess of the destroyed planet Alderaan , who joins the Rebellion ; Luke 's twin sister , and Han 's love interest .
Billy Dee Williams as Lando Calrissian : The former Baron Administrator of Cloud City and one of Han 's friends who aids the Rebellion .
Anthony Daniels as C - 3PO : A humanoid protocol droid , who sides with the Rebellion .
Peter Mayhew as Chewbacca : A Wookiee who is Han 's longtime friend , who takes part in the Rebellion .
Kenny Baker as R2 - D2 : An astromech droid , bought by Luke ; and long - time friend to C - 3PO . He also portrays a GONK power droid in the background .
Ian McDiarmid as the Emperor : The evil founding supreme ruler of the Galactic Empire , and Vader 's Sith Master .
Frank Oz as Yoda : The wise , centuries - old Grand Master of the Jedi , who is Luke 's self - exiled Jedi Master living on Dagobah . After dying , he reappears to Luke as a Force - ghost . Yoda 's Puppetry was assisted by Mike Quinn .
David Prowse as Darth Vader / Anakin Skywalker : A powerful Sith lord and the second in command of the Galactic Empire ; Luke and Leia 's father .
<|text_end|>
Output with 4 QnAs:
<|separator|>

[Q]: who played luke father in return of the jedi
[A]: David Prowse acted as Darth Vader, a.k.a Anakin Skywalker, which is Luke and Leia's father.
[Q]: Who is Han Solo's best friend? And what species is he?
[A]: Han Solo's best friend is Chewbacca, who is a Wookiee.
[Q]: Who played luke's teacher in the return of the jedi
[A]: Yoda, the wise, centuries-old Grand Master of the Jedi, who is Luke's self-exiled Jedi Master living on Dagobah, was played by Frank Oz.
Also, there is a mention of Obi-Wan Kenobi, who trained Luke Skywalker.
But I can't find who played Obi-Wan Kenobi in the given text.
[Q]: Where Yoda lives in the return of the jedi?
[A]: Yoda, the Jedi Master, lives on Dagobah.
<|separator|>

Text:
<|text_start|>
{{text}}
<|text_end|>
Output with {{num_questions}} QnAs:
<|separator|>
"""
QA_TEMPERATURE = 1.0


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
    global RAG_MODEL, CHAT_DOCUMENTS, SEARCH_QUERIES
    res, docs, search_queries = RAG_MODEL.run(input)
    CHAT_DOCUMENTS = json.dumps(docs, indent=4)
    SEARCH_QUERIES = search_queries
    yield res + " "


def navigate_to_eval(event: me.ClickEvent):
    me.navigate("/eval/create")


def navigate_to_config(event: me.ClickEvent):
    me.navigate("/config")


def navigate_to_chat(event: me.ClickEvent):
    me.navigate("/chat")


@me.page(
    path="/chat",
    title="RAG Alligator | Chat",
)
def chat_page():
    global RAG_MODEL, VECTOR_STORE
    if RAG_MODEL is None and VECTOR_STORE is not None:
        print(VECTOR_STORE)
        RAG_MODEL = create_rag_model(VECTOR_STORE)

    with me.box(
        style=me.Style(
            display="flex", flex_direction="row", gap=12, justify_content="center"
        )
    ):
        me.button("chat", disabled=True)
        me.button("eval", on_click=navigate_to_eval)
        me.button("config", on_click=navigate_to_config)

    with me.box(
        style=me.Style(display="flex", width="95vw", flex_direction="row", gap=10)
    ):
        with me.box(style=me.Style(flex_grow=1, flex_basis="30%")):
            mel.chat(
                transform,
                title="RAG Alligator Chat",
                bot_user="Bot",
            )
        if CHAT_DOCUMENTS:
            with me.box(style=me.Style(flex_grow=1, height="95vh")):
                if SEARCH_QUERIES:
                    me.text("Search Queries:", type="body-1")
                    me.native_textarea(
                        value=str(SEARCH_QUERIES),
                        style=me.Style(width="100%", flex_grow=1),
                        readonly=True,
                        autosize=True,
                    )
                me.text("Documents Retrieved for current message:", type="body-1")
                me.native_textarea(
                    value=CHAT_DOCUMENTS,
                    style=me.Style(width="100%", height="95vh", flex_grow=1),
                    readonly=True,
                    autosize=True,
                )


def on_input_qa_data(e: me.InputEvent):
    global QA_SET_JSON
    QA_SET_JSON = e.value


def save_qa_data(event: me.ClickEvent):
    global QA_SET_JSON
    df = pd.read_json(QA_SET_JSON, orient="records")
    df.to_parquet(EVALUATION_SET_PATH)


def set_content_size(event: me.InputEvent):
    global QA_CONTENT_SIZE
    QA_CONTENT_SIZE = event.value


def set_num_questions(event: me.InputEvent):
    global QA_QUESTION_NUM_PER_CONTENT
    QA_QUESTION_NUM_PER_CONTENT = event.value


def set_qa_prompt(event: me.InputEvent):
    global QA_PROMPT
    QA_PROMPT = event.value


def set_qa_temperature(event: me.InputEvent):
    global QA_TEMPERATURE
    QA_TEMPERATURE = event.value


def generate_eval_click(event: me.ClickEvent):
    global QA_SET_JSON, VECTOR_STORE_PATH, QA_CONTENT_SIZE, QA_QUESTION_NUM_PER_CONTENT, QA_TEMPERATURE
    QA_SET_JSON = generate_default_qa_set(
        VECTOR_STORE_PATH,
        int(QA_CONTENT_SIZE),
        int(QA_QUESTION_NUM_PER_CONTENT),
        float(QA_TEMPERATURE),
    )  # will create the parquet file
    print("generating qa set")


@me.page(
    path="/eval/create",
    title="RAG Alligator | Evaluation",
)
def eval_create_page():
    global QA_SET_JSON, CORPUS_JSON

    with me.box(
        style=me.Style(
            display="flex", flex_direction="row", gap=12, justify_content="center"
        )
    ):
        me.button("chat", on_click=navigate_to_chat)
        me.button("eval", disabled=True)
        me.button("config", on_click=navigate_to_config)

    if QA_SET_JSON is None:
        me.text("Evaluation Set Creator", type="headline-4")
        if not os.path.exists(EVALUATION_SET_PATH):
            me.text(f"Content Size: {QA_CONTENT_SIZE}")
            me.input(
                label="Content Size",
                on_input=set_content_size,
                type="number",
            )
            me.text(f"Number of questions per content: {QA_QUESTION_NUM_PER_CONTENT}")
            me.input(
                label="Number of questions per content",
                on_input=set_num_questions,
                type="number",
            )
            me.text("Current QA Prompt")
            me.native_textarea(
                value=QA_PROMPT,
                readonly=True,
                style=me.Style(width="100%", height="100%"),
            )
            me.textarea(
                label="New QA Prompt",
                on_input=set_qa_prompt,
                style=me.Style(width="100%", height="40%"),
            )
            me.text(f"QA Temperature: {QA_TEMPERATURE}")
            me.input(
                label="QA Temperature",
                on_input=set_qa_temperature,
                type="number",
            )
            me.button(
                "Generate Evaluation Set",
                color="primary",
                type="flat",
                on_click=generate_eval_click,
                style=me.Style(margin=me.Margin.all(7), display="block"),
            )
        else:
            qa_set_df = pd.read_parquet(EVALUATION_SET_PATH)
            QA_SET_JSON = qa_set_df.to_json(orient="records", indent=4)
            print("reading qa set")
    if CORPUS_JSON is None:
        corpus_df = pd.read_parquet(VECTOR_STORE_PATH)
        CORPUS_JSON = corpus_df.to_json(orient="records", indent=4)

    if QA_SET_JSON is not None:
        me.text("Evaluation Set", type="headline-4")
        # with me.box(
        #     style=me.Style(
        #         padding=me.Padding.all(10),
        #         height="95vh",
        #     )
        # ):
        me.native_textarea(
            placeholder="Eval Set Data",
            value=QA_SET_JSON,
            style=me.Style(width="100%", height="100%"),
            on_input=on_input_qa_data,
            key="qa_data",
        )
        # me.native_textarea(
        #     placeholder="Vectorized Corpus Data",
        #     value=CORPUS_JSON,
        #     style=me.Style(width="48vw", height="100%"),
        #     readonly=True,
        #     key="qa_corpus",
        # )
        me.button(
            "Save",
            color="primary",
            type="flat",
            on_click=save_qa_data,
            style=me.Style(margin=me.Margin.all(7)),
        )
        me.button(
            "Run Evaluation",
            color="primary",
            type="raised",
            on_click=lambda e: me.navigate("/eval/run"),
            style=me.Style(margin=me.Margin.all(7)),
        )


@me.page(
    path="/eval/run",
    title="RAG Alligator | Evaluation",
)
def eval_run_page():
    global RAGAS_METRICS

    with me.box(
        style=me.Style(
            display="flex", flex_direction="row", gap=12, justify_content="center"
        )
    ):
        me.button("chat", on_click=navigate_to_chat)
        me.button("eval", disabled=True)
        me.button("config", on_click=navigate_to_config)

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

    with me.box(
        style=me.Style(
            display="flex", flex_direction="row", gap=12, justify_content="center"
        )
    ):
        me.button("chat", on_click=navigate_to_chat)
        me.button("eval", on_click=navigate_to_eval)
        me.button("config", disabled=True)

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

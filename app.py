from typing import List

import mesop as me
import mesop.labs as mel

from page_functions.vectorstore import create_vectorstore


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
    state = me.state(State)
    state.vectorstore = create_vectorstore(state.urls)
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


@me.page(
    path="/chat",
    title="RAG Alligator | Chat",
)
def chat_page():
    me.text(text="Chat Page")
    # mel.chat(transform, title="RAG Alligator Chat", bot_user="Bot")


@me.page(
    path="/eval/create",
    title="RAG Alligator | Evaluation",
)
def eval_create_page():
    pass


@me.page(
    path="/eval/run",
    title="RAG Alligator | Evaluation",
)
def eval_run_page():
    pass

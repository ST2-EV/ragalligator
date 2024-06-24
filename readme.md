# RAG Alligator - Alpha

A simple ui tool to get a baseline model and evaluation set for a RAG application.

PS: Currently only supports webpages to be used as a knowledgebase and UI is barebones.

#### Motivation

It's very easy to build an impressive looking demo but hard to build a useful application that gets used in production. In my experience, the domain expert and engineer are always different people so I wanted something that could harmonize their efforts. So RAG Alligator's goal is to solve that disconnect by allowing the domain expert to build out an evaluation set and a baseline model very quickly with no code and identify a target value for the for final model to achieve on the created evaluation set.

#### Demo

# Installation

```
git clone
cd ragalligator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Run

```
mesop app.py
```

# Todo

- [ ] Remove repition bug in Eval creation tool
- [ ] Add custom component based ui for eval creation tool
- [ ] Show a more detailed progress bar when running the evaluation set
- [ ] Add option to use pdfs as knowledgebase
- [ ] Add option to use docx as knowledgebase
- [ ] Add option to use googledrive as knowledgebase

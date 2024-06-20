import os
import uuid

import cohere

co = cohere.Client(os.getenv("COHERE_API_KEY"))


class Chatbot:
    def __init__(self, vectorstore):
        """
        Initializes an instance of the Chatbot class.

        Parameters:
        vectorstore (Vectorstore): An instance of the Vectorstore class.

        """
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())

    def run(self, message):
        """
        Runs the chatbot application.

        """
        response = co.chat(message=message, search_queries_only=True)
        if response.search_queries:
            print("Retrieving information...", end="")

            # Retrieve document chunks for each query
            documents = []
            for query in response.search_queries:
                documents.extend(self.vectorstore.retrieve(query.text))

            # Use document chunks to respond
            response = co.chat_stream(
                message=message,
                model="command-r",
                documents=documents,
                conversation_id=self.conversation_id,
            )

        # If there is no search query, directly respond
        else:
            response = co.chat_stream(
                message=message,
                model="command-r",
                conversation_id=self.conversation_id,
            )

        citations = []
        cited_documents = []

        response_str = []
        # Display response
        for event in response:
            if event.event_type == "text-generation":
                response_str.append(event.text)
            elif event.event_type == "citation-generation":
                citations.extend(event.citations)
            elif event.event_type == "search-results":
                cited_documents = event.documents

        return " ".join(response_str)


def create_rag_model(vectorstore):
    return Chatbot(vectorstore)

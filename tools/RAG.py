import os
import urllib.request
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack import component
from dotenv import load_dotenv

load_dotenv()

# urllib.request.urlretrieve(
#     "https://www.gutenberg.org/cache/epub/2600/pg2600.txt", "WAR_and_PEACE.txt"
# )

# document_store = InMemoryDocumentStore()

# text_file_converter = TextFileToDocument()
# cleaner = DocumentCleaner()
# splitter = DocumentSplitter()
# embedder = OpenAIDocumentEmbedder()
# writer = DocumentWriter(document_store)

# indexing_pipeline = Pipeline()
# indexing_pipeline.add_component("converter", text_file_converter)
# indexing_pipeline.add_component("cleaner", cleaner)
# indexing_pipeline.add_component("splitter", splitter)
# indexing_pipeline.add_component("embedder", embedder)
# indexing_pipeline.add_component("writer", writer)

# indexing_pipeline.connect("converter.documents", "cleaner.documents")
# indexing_pipeline.connect("cleaner.documents", "splitter.documents")
# indexing_pipeline.connect("splitter.documents", "embedder.documents")
# indexing_pipeline.connect("embedder.documents", "writer.documents")
# indexing_pipeline.run(data={"sources": ["davinci.txt"]})

# text_embedder = OpenAITextEmbedder()
# retriever = InMemoryEmbeddingRetriever(document_store)
# template = """Given these documents, answer the question.
#               Documents:
#               {% for doc in documents %}
#                   {{ doc.content }}
#               {% endfor %}
#               Question: {{query}}
#               Answer:"""
# prompt_builder = PromptBuilder(template=template)
# llm = OpenAIGenerator()

# rag_pipeline = Pipeline()
# rag_pipeline.add_component("text_embedder", text_embedder)
# rag_pipeline.add_component("retriever", retriever)
# rag_pipeline.add_component("prompt_builder", prompt_builder)
# rag_pipeline.add_component("llm", llm)

# rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
# rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
# rag_pipeline.connect("prompt_builder", "llm")

# query = "How old was Leonardo when he died?"
# result = rag_pipeline.run(
#     data={"prompt_builder": {"query": query}, "text_embedder": {"text": query}}
# )
# print(result["llm"]["replies"][0])


@component
class RAG:
    def __init__(self, url: str, name: str):
        urllib.request.urlretrieve(url, name)

        document_store = InMemoryDocumentStore()

        # text_file_converter = TextFileToDocument()
        # cleaner = DocumentCleaner()
        # splitter = DocumentSplitter()
        # embedder = OpenAIDocumentEmbedder()
        # writer = DocumentWriter(document_store)

        # indexing_pipeline = Pipeline()
        # indexing_pipeline.add_component("converter", text_file_converter)
        # indexing_pipeline.add_component("cleaner", cleaner)
        # indexing_pipeline.add_component("splitter", splitter)
        # indexing_pipeline.add_component("embedder", embedder)
        # indexing_pipeline.add_component("writer", writer)

        # indexing_pipeline.connect("converter.documents", "cleaner.documents")
        # indexing_pipeline.connect("cleaner.documents", "splitter.documents")
        # indexing_pipeline.connect("splitter.documents", "embedder.documents")
        # indexing_pipeline.connect("embedder.documents", "writer.documents")
        # indexing_pipeline.run(data={"sources": ["WAR_and_PEACE.txt"]})

        text_embedder = OpenAITextEmbedder()
        retriever = InMemoryEmbeddingRetriever(document_store)
        template = """Given these documents, answer the question.
                    Documents:
                    {% for doc in documents %}
                        {{ doc.content }}
                    {% endfor %}
                    Question: {{query}}
                    Answer:"""
        prompt_builder = PromptBuilder(template=template)
        llm = OpenAIGenerator()

        rag_pipeline = Pipeline()
        rag_pipeline.add_component("text_embedder", text_embedder)
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("llm", llm)

        rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm")

        self.RAG = rag_pipeline

    @component.output_types(result=str)
    def run(self, query: str):
        result = self.RAG.run(
            data={"prompt_builder": {"query": query}, "text_embedder": {"text": query}}
        )
        return {"result": result["llm"]["replies"][0]}

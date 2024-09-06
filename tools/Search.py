from langchain_community.tools.tavily_search import TavilySearchResults
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

tavilytool = TavilySearchResults()


@component
class TavilySearch:
    def __init__(self):
        self.tavilytool = tavilytool

    @component.output_types(result=str)
    def run(self, query: str) -> str:
        return {"result": self.tavilytool.invoke(query)}

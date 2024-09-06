from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators.openai import OpenAIGenerator
from typing import List
from tools.RAG import RAG
from tools.Search import TavilySearch
from tools.Evaluator import Evaluate
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.routers import ConditionalRouter
import weave

weave.init("haystack")

prompt = PromptBuilder(
    template="""
    You are an AI Assistant. Given the query you should decide which route to choose.
    Query: {{query}}
    Evaluation:{{evaluation}}
    Agents: {{agents}}
    Answer:
"""
)


routes = [
    {
        "condition": "{{'RAG' in replies[0]}}",
        "output": "{{query}}",
        "output_name": "RAG",
        "output_type": str,
    },
    {
        "condition": "{{'Search' in replies[0]}}",
        "output": "{{query}}",
        "output_name": "Search",
        "output_type": str,
    },
    {
        "condition": "{{'Search' not in replies[0] and 'RAG' not in replies[0]}}",
        "output": "{{replies[0]}}",
        "output_name": "answer",
        "output_type": str,
    },
]
router = ConditionalRouter(routes)

route2 = [
    {
        "condition": "{{'__end__' in evaluation}}",
        "output": "{{result}}",
        "output_name": "result",
        "output_type": str,
    },
    {
        "condition": "{{'__end__' not in evaluation}}",
        "output": "{{evaluation}}",
        "output_name": "go_back",
        "output_type": str,
    },
]
router2 = ConditionalRouter(route2)

route3 = [
    {
        "condition": "{{'__end__' in evaluation}}",
        "output": "{{result}}",
        "output_name": "result",
        "output_type": str,
    },
    {
        "condition": "{{'__end__' not in evaluation}}",
        "output": "{{evaluation}}",
        "output_name": "go_back",
        "output_type": str,
    },
]
router3 = ConditionalRouter(route3)

RAG = RAG("https://www.gutenberg.org/cache/epub/2600/pg2600.txt", "WAR_and_Peace.txt")
# print(RAG.run("What's Prince Andrey's best friend's wife's name?"))

Evaluator1 = Evaluate()
Evaluator2 = Evaluate()

fallback_prompt = PromptBuilder(
    template="""
    Answer is {{answer}}.
"""
)
fallback_prompt2 = PromptBuilder(
    template="""
    Answer is {{answer}}.
"""
)
fallback_prompt3 = PromptBuilder(
    template="""
    Answer is {{answer}}.
"""
)

search = TavilySearch()
llm = OpenAIGenerator(model="gpt-4o")

components = ["RAG", "Search", "fallback_prompt"]
pipeline = Pipeline()
pipeline.add_component("prompt", prompt)
pipeline.add_component("router", router)
pipeline.add_component("RAG", RAG)
pipeline.add_component("Search", search)
pipeline.add_component("llm", llm)
pipeline.add_component("fallback_prompt", fallback_prompt)
pipeline.add_component("router2", router2)
pipeline.add_component("router3", router3)
pipeline.add_component("Evaluate1", Evaluator1)
pipeline.add_component("Evaluate2", Evaluator2)
pipeline.add_component("fallback_prompt2", fallback_prompt2)
pipeline.add_component("fallback_prompt3", fallback_prompt3)

pipeline.connect("prompt", "llm")
# pipeline.connect("prompt.prompt", "router.query")
pipeline.connect("llm.replies", "router.replies")
pipeline.connect("router.RAG", "RAG.query")
pipeline.connect("router.Search", "Search.query")
pipeline.connect("router.answer", "fallback_prompt.answer")
pipeline.connect("RAG.result", "Evaluate1.result")
pipeline.connect("Search.result", "Evaluate2.result")
pipeline.connect("Evaluate1.result", "router2.result")
pipeline.connect("Evaluate1.evaluation", "router2.evaluation")
pipeline.connect("Evaluate2.result", "router3.result")
pipeline.connect("Evaluate2.evaluation", "router3.evaluation")
pipeline.connect("router2.result", "fallback_prompt2.answer")
pipeline.connect("router2.go_back", "prompt.evaluation")
pipeline.connect("router3.result", "fallback_prompt3.answer")
# pipeline.connect("router3.go_back", "prompt.evaluation")

pipeline.draw("pipeline.png")

query = "Give detailed description of Prince Andrey's wife death moments."


@weave.op()
def check():
    ans = pipeline.run(
        {
            "router": {
                "query": query,
            },
            "prompt": {
                "query": query,
                "agents": components,
            },
            "Evaluate1": {
                "query": query,
            },
            "Evaluate2": {
                "query": query,
            },
        }
    )
    print(ans)


check()

from haystack import component


@component
class Evaluate:
    def __init__(self):
        pass

    @component.output_types(evaluation=str, result=str)
    def run(self, query: str, result: str) -> str:
        return {"result": result, "evaluation": "__end__"}

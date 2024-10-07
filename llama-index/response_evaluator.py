import asyncio
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.evaluation import DatasetGenerator


def generate_questions(documents):
    question_generator = DatasetGenerator.from_documents(documents)
    return question_generator.generate_questions_from_nodes(5)


def build_evaluator(llm):
    return FaithfulnessEvaluator(llm)


def evaluate_query_engine(query_engine, questions, evaluator):
    c = [query_engine.aquery(q) for q in questions]
    results = asyncio.run(asyncio.gather(*c))
    print("finished query")

    total_correct = 0
    for r in results:
        # evaluate with gpt 4
        eval_result = 1 if evaluator.evaluate_response(response=r).passing else 0
        total_correct += eval_result

    return total_correct, len(results)

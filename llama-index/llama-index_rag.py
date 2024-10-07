import weaviate
import argparse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import FLAREInstructQueryEngine
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.core.llama_dataset.generator import RagDatasetGenerator

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.metrics.critique import harmfulness
from ragas.integrations.llama_index import evaluate
from llama_index.embeddings.openai import OpenAIEmbedding

qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query. If you cant find the answer in the context information,"
    "just say you don't know the answer\n"
    ""
    "Query: {query_str}\n"
    "Answer: "
)


class RAGQueryEngine(CustomQueryEngine):

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)

        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = Settings.llm.complete(
            qa_prompt.format(context_str=context_str, query_str=query_str)
        )

        return str(response)


def menu():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", help="Query string")
    parser.add_argument(
        "-e",
        "--evaluate_querying",
        action="store_true",
        help="Evaluate the answers with generated questions from the vector store",
    )
    parser.add_argument(
        "-b",
        "--build_vector",
        action="store_true",
        help="Build vector store from Documents",
    )
    parser.add_argument("--documents_path", help="Path to the documents folder")
    args = parser.parse_args()

    return args


def test_generator(documents):
    embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    generator = TestsetGenerator.from_llama_index(
        generator_llm=Settings.llm,
        critic_llm=Settings.llm,
        embeddings=embeddings,
    )
    print(generator)
    # generate testset
    testset = generator.generate_with_llamaindex_docs(
        documents,
        test_size=1,
        distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
    )
    return testset.to_pandas()


def load_models():
    Settings.llm = Ollama(model="llama2", request_timeout=600)
    # Settings.critic_llm = Ollama(model="llama2", request_timeout=600)
    # Settings.generator_llm = Ollama(model="llama2", request_timeout=600)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def load_files(files_path: str, num_files_limit: int = 3) -> list:
    documents = SimpleDirectoryReader(
        input_dir=files_path, num_files_limit=1
    ).load_data()
    return documents


def build_node_parser(documents: list) -> list:
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=5,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
        include_prev_next_rel=True,
    )

    # extracting nodes from documents
    nodes = node_parser.get_nodes_from_documents(documents)
    return nodes


def start_weaviate_client() -> weaviate.client.Client:
    client = weaviate.Client(embedded_options=weaviate.embedded.EmbeddedOptions())
    print("Client is ready: ", client.is_ready())
    return client


def build_vector_store(nodes: list, client: weaviate.client.Client, index_name: str):
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if client.schema.exists(index_name):
        client.schema.delete_class(index_name)

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
    )
    return index


def load_vector_store(index_name: str, client: weaviate.client.Client):
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)
    loaded_index = VectorStoreIndex.from_vector_store(vector_store)
    return loaded_index


def build_query_engine(loaded_index):
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base")

    query_engine = loaded_index.as_query_engine(
        similarity_top_k=6,
        vector_store_query_mode="hybrid",
        alpha=0.5,
        node_postprocessor=[postproc, rerank],
    )
    return query_engine


def build_custom_query_engine(loaded_index):
    retriever = loaded_index.as_retriever()
    synthesizer = get_response_synthesizer(response_mode="compact")

    query_engine = RAGQueryEngine(
        retriever=retriever, response_synthesizer=synthesizer, qa_prompt=qa_prompt
    )
    return query_engine


def run_query_flare(query_engine, query: str):
    flare_query_engine = FLAREInstructQueryEngine(
        query_engine=query_engine, max_iterations=2, verbose=True
    )
    query_result = flare_query_engine.query(query)
    return query_result


def run_query(query_engine, query: str):
    query_result = query_engine.query(query)
    return query_result


def evaluate_model(testset, query_engine):
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        harmfulness,
    ]
    results = evaluate(
        query_engine=query_engine,
        metrics=metrics,
        dataset=testset,
        llm=Settings.llm,
        embeddings=Settings.embed_model,
    )
    results.to_pandas().to_csv("results.csv")


def main():
    options = menu()
    load_models()
    client = start_weaviate_client()

    if options.build_vector:
        print("Building vector store from documents...\n")
        documents = load_files(options.documents_path, num_files_limit=3)
        nodes = build_node_parser(documents)
        index = build_vector_store(nodes=nodes, client=client, index_name="NewIndex")
    else:
        print("Loading vector store from Weaviate...\n")
        index = load_vector_store("NewIndex", client)

    if options.evaluate_querying:
        # query_engine = build_query_engine(index)
        # testset = test_generator(load_files(options.documents_path))
        # evaluate_model(testset, query_engine)
        evaluator_faith = FaithfulnessEvaluator(llm=Settings.llm)
        evaluator_relevance = RelevancyEvaluator(llm=Settings.llm)

    if options.query is not None:
        query_engine = build_query_engine(index)
        print("Querying...")
        response = run_query(query_engine, options.query)

        eval_result_f = evaluator_faith.evaluate_response(
            response=response, query=options.query
        )
        eval_result_r = evaluator_relevance.evaluate_response(
            response=response, query=options.query
        )
        print("\n", eval_result_f)
        print("\n", eval_result_r)
        return


if __name__ == "__main__":
    main()

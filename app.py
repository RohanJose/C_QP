from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import faiss
import pickle
import torch
import gradio as gd

from tqdm import tqdm
from langchain.docstore import InMemoryDocstore
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from ragatouille import RAGPretrainedModel
from typing import List, Tuple, Optional
from langchain.docstore.document import Document as LangchainDocument


EMBEDDING_MODEL_NAME = "thenlper/gte-small"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
index = faiss.read_index('knowledge_vector_1.index')
with open('docs_processed.pkl', 'rb') as f:
    docs_processed = pickle.load(f)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

docstore = InMemoryDocstore({i: doc for i, doc in enumerate(docs_processed)})

KNOWLEDGE_VECTOR_DATABASE = FAISS(
    index=index,
    docstore=docstore,
    index_to_docstore_id={i: i for i in range(len(docs_processed))},
    embedding_function=embedding_model)

# Model initialization
READER_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=1000,
)

# Initialize reranker
RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

prompt_in_chat_format = [
    {
        "role": "system",
        "content": """You are an AI assistant specializing in analyzing PDF documents. Your task is to generate a comprehensive question paper based on the provided PDF context.
        For each section mentioned, generate the exact number of questions as specified.
        Ensure that the questions are relevant, clear, and cover the key topics within the section.
        Reference specific page numbers or sections from the PDF whenever applicable.
        If the information needed to create questions is not available in the PDF context, clearly state that.
        """,
    },
    {
        "role": "user",
        "content": """PDF Context:
        {context}
        ---
        For the following sections, generate new C programming questions based on the context the required number of questions. Provide a header also:
        section_requirements
    total marks:100
    part A-10 questions ,
    part B- 5 questions,
    part C- 4 questions,
    1 compulsory question

        ---
        Question: {question}""",
    },
]

RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
    prompt_in_chat_format, tokenize=False, add_generation_prompt=True
)

def answer_with_rag(
    question: str,
    llm: pipeline,
    knowledge_index: FAISS,
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 5,
) -> Tuple[str, List[str]]:
    # Gather documents with retriever
    relevant_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text

    # Optionally rerank results
    if reranker:
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]
        relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted PDF content:\n"
    context += "".join([f"Section {str(i+1)}:::\n" + doc for i, doc in enumerate(relevant_docs)])
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    # Generate an answer
    answer = llm(final_prompt)[0]["generated_text"]
    return answer, relevant_docs

def generate_questions(context: str):
    question = "generate end-sem question paper?"
    answer, relevant_docs = answer_with_rag(question, READER_LLM, KNOWLEDGE_VECTOR_DATABASE, reranker=RERANKER)
    return answer

# Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("""
    # C Question Paper Generator

    """)

    with gr.Row():
        context_input = gr.Textbox(label="Enter Prompt", placeholder="prompt", lines=1)

    generate_button = gr.Button("Generate Questions")
    output_text = gr.Textbox(label="Generated Questions", lines=20)

    generate_button.click(generate_questions, inputs=[context_input], outputs=[output_text])

interface.launch()

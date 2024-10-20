from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from unstructured.partition.auto import partition
import os
import pandas as pd
import gradio as gr

def preprocess(folder_path = r'C:\Users\admin\Documents\LLM\B1-B data'):
    # 1. Document Loading and Page Tracking
    docs = []
    doc_folder = folder_path
    for filename in os.listdir(doc_folder):
        filepath = os.path.join(doc_folder, filename)
        if os.path.isfile(filepath):
            elements = partition(filename=filepath)
            for i, element in enumerate(elements):
                text = str(element)
                page_number = element.metadata.page_number if element.metadata.page_number else 'N/A'  # Extract page info
                docs.append({"source": filename, "content": text, "page": page_number})
    
    # 2. Chunking while Preserving Page Information
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=300)
    all_splits = []
    current_chunk = ""
    current_metadata = {}
    
    for doc in docs:
        splits = text_splitter.split_text(doc['content'])
        for split in splits:
            if len(current_chunk) + len(split) <= 4096: 
                current_chunk += split + " " 
                current_metadata = {"source": doc['source'], "page": doc['page']} 
            else:
                all_splits.append(Document(page_content=current_chunk, metadata=current_metadata))
                current_chunk = split + " "
                current_metadata = {"source": doc['source'], "page": doc['page']}
    
    if current_chunk:
        all_splits.append(Document(page_content=current_chunk, metadata=current_metadata)) 
    
    # 3. Vectorstore and LLM Setup - Load LLM and Vectorstore only once
    model = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=model)
    llm = ChatOllama(model="llama3.1:8b")  # LLM loaded only once
    return llm, vectorstore

# 4. RAG Function (no LLM loading inside)
def RAG(user_prompt, llm, vectorstore, stream=False, source_summaries=False, retrieval='contextual', top_k_hits=5):
    def format_docs(docs):
        return "\n\n".join(
            f"Source: {doc.metadata['source']} - Page: {doc.metadata.get('page', 'N/A')}\n\n{doc.page_content}" 
            for doc in docs
        )

    RAG_TEMPLATE = """
        This is a chat between a user and an artificial intelligence assistant. 
        The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. 
        Cite the source document and page number in parentheses where you found the relevant information.

        <context>
        {context}
        </context>

        Answer the following question:

        {question}"""

    question = user_prompt
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    retriever = vectorstore.as_retriever()

    if retrieval == 'contextual' or retrieval == 'both':
        compressor = LLMChainExtractor.from_llm(llm) 
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        qa_chain = (
            {"context": compression_retriever | format_docs, "question": RunnablePassthrough()} 
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        docs = compression_retriever.invoke(question)  # Invoke on the question 

    if retrieval == 'cosine similarity' or retrieval == 'both':
        if retrieval != 'both':
            qa_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | rag_prompt
                | llm
                | StrOutputParser()
            )
        if retrieval == 'both':
            docs.extend(vectorstore.similarity_search(question, k=top_k_hits))
        else:
            docs = vectorstore.similarity_search(question, k=top_k_hits)

    if not docs:
        return "No relevant documents found", pd.DataFrame()

    source_data = []
    for doc in docs:
        source_data.append({
            "source": doc.metadata['source'], 
            "end page": doc.metadata.get('page', 'N/A'),
            "content": doc.page_content 
        })

    if source_summaries:
        summaries = [llm.invoke(f'Summarize this: <{doc.page_content}> ').content for doc in docs]
        source_df = pd.DataFrame(source_data)
        source_df["short summary"] = summaries
    else:
        source_df = pd.DataFrame(source_data)

    if stream:
        for chunk in qa_chain.stream(question):
            print(chunk, end="", flush=True)
        return '', source_df
    else:
        result = qa_chain.invoke(question)
        return result, source_df

def RAG_gradio(user_prompt, retrieval_method, top_k_hits=5): 
    result, sources_df = RAG(user_prompt, llm, vectorstore, retrieval=retrieval_method, top_k_hits=top_k_hits)
    global folder_path 
    # Format source information for display
    root_dir = folder_path
    sources_df['source'] = sources_df['source'].apply(lambda x: root_dir + '\\' + x)
    # Return the result and HTML representation of the DataFrame
    return result, sources_df.to_html(escape=False)

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("### B-1B Database Q&A")

    with gr.Row():
        user_input = gr.Textbox(label="Enter your question:")
        retrieval_choice = gr.Radio(
            choices=["contextual", "cosine similarity", "both"],
            label="Retrieval Method:",
            value="cosine similarity"
        )
        
        top_k_slider = gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Cosine Similarity Number of Documents:")
        
    output = gr.Markdown(label="Answer:")  
    source_output = gr.HTML(label='Source Information')

    def update_slider(method):
        if method == "contextual":
            return gr.update(visible=True, value=1), gr.update(label="Not Applicable to Contextual Retrieval") 
        elif method == "cosine similarity":
            return gr.update(visible=True, value=5), gr.update(label="Cosine Similarity Number of Documents:") 
        elif method == "both":
            return gr.update(visible=True, value=5), gr.update(label="Cosine Similarity Number of Documents (not contextual):") 

    retrieval_choice.change(update_slider, inputs=retrieval_choice, outputs=[top_k_slider, top_k_slider])

    def show_thinking():
        return "Processing your request..."

    btn = gr.Button("Submit")
    btn.click(fn=show_thinking, inputs=None, outputs=output)
    btn.click(fn=RAG_gradio, inputs=[user_input, 
                                     retrieval_choice, 
                                     top_k_slider], outputs=[output, source_output])

# do not omit the try-except, this allows to create the vectorstore and read in the llm only once. 
try: llm
except: 
    folder_path = r'C:\Users\admin\Documents\LLM\B1-B data'
    llm, vectorstore = preprocess(folder_path)
    
demo.launch()


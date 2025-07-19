import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain

# Load env and initialize Gemini model
load_dotenv()
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# Global memory store per user
memory_store = {}

def get_or_create_chat_chain(user_id):
    if user_id not in memory_store:
        memory = ConversationSummaryMemory(llm=model)
        chain = ConversationChain(llm=model, memory=memory, verbose=False)
        memory_store[user_id] = chain
    return memory_store[user_id]

# Step 1: Load and split text file into chunks
def load_and_split_text(file_path):
    loader = TextLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(docs)

# Step 2: Create retriever from chunks
def create_retriever(chunks, top_k=3):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": top_k})

# Step 3: Rephrase user question into 3 diverse search queries
def rephrase_question(original_question):
    prompt = f"""
    Rephrase the question below into 3 semantically different alternatives
    that might better match relevant information in a vector database.

    Original question:
    "{original_question}"

    Return only the 3 alternative questions in a numbered list.
    """
    response = model.invoke(prompt)
    return [line.strip("0123456789. ").strip() for line in response.content.strip().split("\n") if line.strip()]

# Step 4: Retrieve top matching chunks for each rephrased query
def search_chunks(rephrased_questions, retriever):
    all_docs = []
    for q in rephrased_questions:
        docs = retriever.invoke(q)
        all_docs.extend(docs)
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    return list(unique_docs)

# Step 5: Answer original question using retrieved context
def answer_with_context(original_question, context_chunks):
    context = "\n\n".join([doc.page_content for doc in context_chunks])
    prompt = f"""
    You are a helpful assistant. Use only the context below to answer the user's original question.

    CONTEXT:
    {context}

    USER QUESTION:
    {original_question}

    If the context does not provide a clear answer, say "I don't know" rather than making up an answer.
    """
    response = model.invoke(prompt)
    return response.content

# Step 6: Classify if input is small talk or business question
def classify_intent(user_input):
    prompt = f"""
    Determine if the user input is a casual small talk or a business-related question.

    User input:
    "{user_input}"

    Respond only with one word: "chat" or "business".
    """
    response = model.invoke(prompt)
    return response.content.strip().lower()

# Step 7: Full RAG flow if classified as business
def orchestrate_chat(file_path, user_question):
    chunks = load_and_split_text(file_path)
    retriever = create_retriever(chunks)
    rephrased_questions = rephrase_question(user_question)
    context_chunks = search_chunks(rephrased_questions, retriever)
    return answer_with_context(user_question, context_chunks)

# Example usage
if __name__ == "__main__":
    user_question = "what time are you open?"
    file_path = "company_doc.txt"
    user_id = "12345"  # This should be dynamically passed (e.g., Telegram user ID)

    intent = classify_intent(user_question)

    if intent == "business":
        final_answer = orchestrate_chat(file_path, user_question)
    else:
        chain = get_or_create_chat_chain(user_id)
        final_answer = chain.predict(input=user_question)

    print("\nFinal Answer:\n")
    print(final_answer)

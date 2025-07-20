from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Initialize Gemini model
print("[INIT] Initializing Gemini model...", flush=True)
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# Per-user memory storage
memory_store = {}

from langchain_core.prompts import MessagesPlaceholder

def get_or_create_chat_chain(user_id: str):
    print(f"[MEMORY] Getting or creating chat memory for user_id={user_id}", flush=True)

    def get_history(session_id: str):
        print(f"[MEMORY] Retrieving history for session_id={session_id}", flush=True)
        if session_id not in memory_store:
            print(f"[MEMORY] No history found. Creating new memory.", flush=True)
            memory_store[session_id] = InMemoryChatMessageHistory()
        return memory_store[session_id]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}")
    ])

    chain = prompt | model

    return RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="input",
        history_messages_key="messages"
    )


# Step 1: Load and split text file into chunks
def load_and_split_text(file_path):
    print(f"[LOAD] Loading document from {file_path}", flush=True)
    loader = TextLoader(file_path)
    docs = loader.load()
    print(f"[LOAD] Loaded {len(docs)} document(s)", flush=True)

    if not docs:
        print("[ERROR] No content loaded from file!", flush=True)
        return []

    total_chars = sum(len(doc.page_content) for doc in docs)
    print(f"[LOAD] Total characters: {total_chars}", flush=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"[SPLIT] Split into {len(chunks)} chunk(s)", flush=True)
    return chunks

# Step 2: Create retriever from chunks
def create_retriever(chunks, top_k=3):
    print(f"[RETRIEVER] Creating retriever with top_k={top_k}", flush=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": top_k})

# Step 3: Rephrase user question into 3 diverse search queries
def rephrase_question(original_question):
    print(f"[REPHRASE] Rephrasing question: {original_question}", flush=True)
    prompt = f"""
    Rephrase the question below into 3 semantically different alternatives
    that might better match relevant information in a vector database.

    Original question:
    "{original_question}"

    Return only the 3 alternative questions in a numbered list.
    """
    response = model.invoke(prompt)
    print(f"[REPHRASE] Raw response:\n{response.content}", flush=True)

    lines = response.content.strip().split("\n")

    rephrased = []
    for line in lines:
        cleaned = line.strip().lstrip("0123456789. ").strip()
        if cleaned:
            rephrased.append(cleaned)
        if len(rephrased) == 3:
            break

    while len(rephrased) < 3:
        print(f"[REPHRASE] Filling missing rephrased question with original...", flush=True)
        rephrased.append(original_question)

    print(f"[REPHRASE] Final rephrased list: {rephrased}", flush=True)
    return rephrased

# Step 4: Retrieve top matching chunks for each rephrased query
def search_chunks(rephrased_questions, retriever):
    print(f"[SEARCH] Searching chunks for rephrased questions...", flush=True)
    all_docs = []
    for q in rephrased_questions:
        print(f"[SEARCH] Query: {q}", flush=True)
        docs = retriever.invoke(q)
        print(f"[SEARCH] Found {len(docs)} docs for this query", flush=True)
        all_docs.extend(docs)

    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    print(f"[SEARCH] Deduplicated to {len(unique_docs)} unique docs", flush=True)
    return list(unique_docs)

# Step 5: Answer original question using retrieved context
def answer_with_context(original_question, context_chunks):
    print(f"[ANSWER] Answering question using context...", flush=True)
    context = "\n\n".join([doc.page_content for doc in context_chunks])
    print(f"[ANSWER] Context preview:\n{context[:500]}...", flush=True)

    prompt = f"""
    You are a helpful assistant. Use only the context below to answer the user's original question.

    CONTEXT:
    {context}

    USER QUESTION:
    {original_question}

    If the context does not provide a clear answer, say "I don't know" rather than making up an answer.
    """
    response = model.invoke(prompt)
    print(f"[ANSWER] Raw response:\n{response.content}", flush=True)
    return response.content

# Step 6: Classify if input is small talk or business question
def classify_intent(user_input):
    print(f"[CLASSIFY] Classifying user input:\n{user_input}", flush=True)
    prompt = f"""
    Determine if the user input is a casual small talk or a business-related question.

    User input:
    "{user_input}"

    Respond only with one word: "chat" or "business".
    """
    response = model.invoke(prompt)
    print(f"[CLASSIFY] Classification result: {response.content.strip()}", flush=True)
    return response.content.strip().lower()

# Step 7: Full RAG flow if classified as business
def orchestrate_chat(file_path, user_question):
    print("[ORCHESTRATE] Running full RAG flow...", flush=True)
    chunks = load_and_split_text(file_path)

    if not chunks:
        print("[ERROR] No chunks to search. Returning fallback message.", flush=True)
        return "⚠️ I couldn't find any relevant information in the knowledge file."

    retriever = create_retriever(chunks)
    rephrased_questions = rephrase_question(user_question)
    print(f"[RERANK] Rephrased into: {rephrased_questions}", flush=True)

    context_chunks = search_chunks(rephrased_questions, retriever)
    print(f"[RETRIEVE] Retrieved {len(context_chunks)} unique chunks", flush=True)

    return answer_with_context(user_question, context_chunks)


# Test-only
#if __name__ == "__main__":
#    user_question = "what are your AI services?"
#    user_id = "12345"
#    doc_path = "uploads/company_doc.txt"
#
#    print(f"\n[TEST] Testing with question: {user_question}")
#
#    intent = classify_intent(user_question)
#    print(f"[TEST] Intent classified as: {intent}")
#
#    if intent == "business":
#        answer = orchestrate_chat(doc_path, user_question)
#    else:
#        chain = get_or_create_chat_chain(user_id)
#        print("[TEST] Using chat memory chain...")
#        answer = chain.invoke({"input": user_question}, config={"configurable": {"session_id": user_id}})
#
#    print("\n[FINAL ANSWER]\n", answer)
#
import os
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from openai import OpenAI
from langchain_core.documents import Document
from flask_cors import CORS
import os.path

app = Flask(__name__)
CORS(app)

try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=openai_api_key)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
except Exception as e:
    print(f"Error initializing OpenAI services: {str(e)}")
    exit(1)

def initialize_cccu_database():
    try:
        persist_directory = "./chroma_db"
        
        if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
            print("Loading existing Chroma database from disk...")
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        else:
            print("Creating new Chroma database...")
            cccu_general_data = [
                Document(
                    page_content="Canterbury Christ Church University (CCCU) was founded in 1962 and received university status in 2005.",
                    metadata={"source": "general_info", "category": "history"}
                ),
                Document(
                    page_content="CCCU has approximately 15,000 students across its campuses.",
                    metadata={"source": "general_info", "category": "statistics"}
                ),
                Document(
                    page_content="Canterbury Christ Church University has multiple campuses in Canterbury, Medway, and Tunbridge Wells.",
                    metadata={"source": "general_info", "category": "locations"}
                ),
                Document(
                    page_content="CCCU is known for education, health, and arts programs.",
                    metadata={"source": "general_info", "category": "programs"}
                ),
                Document(
                    page_content="The university was awarded Silver in the Teaching Excellence Framework (TEF).",
                    metadata={"source": "general_info", "category": "awards"}
                ),
                Document(
                    page_content="In UK rankings, CCCU is typically in the 90-110 range among UK universities.",
                    metadata={"source": "general_info", "category": "rankings"}
                ),
                Document(
                    page_content="International student tuition fees at CCCU range from £14,500 to £16,000 per year for undergraduate programs.",
                    metadata={"source": "general_info", "category": "international_fees"}
                ),
                Document(
                    page_content="CCCU offers various scholarships and bursaries for international students.",
                    metadata={"source": "general_info", "category": "international_support"}
                ),
                Document(
                    page_content="International students at CCCU come from over 85 countries worldwide.",
                    metadata={"source": "general_info", "category": "international_community"}
                )
            ]
            
            cccu_menu_data = [
                Document(
                    page_content="Touchdown Cafe at CCCU offers a range of beverages including: Cappuccino (£1.95), Latte (£1.95), Spiced Chai (£1.95), Vanilla Chai (£1.95), Coke (£1.65), and Lipton Ice Lemon Tea (£1.50).",
                    metadata={"source": "menu_info", "category": "touchdown_cafe", "type": "beverages"}
                ),
                Document(
                    page_content="Touchdown Cafe at CCCU offers food items including: Bacon and Cheese (£1.95) and Chocolate Cake (£2.50).",
                    metadata={"source": "menu_info", "category": "touchdown_cafe", "type": "food"}
                ),
                Document(
                    page_content="Touchdown Cafe is one of the dining options available at Canterbury Christ Church University.",
                    metadata={"source": "menu_info", "category": "touchdown_cafe", "type": "general"}
                )
            ]
            
            all_documents = cccu_general_data + cccu_menu_data
            vectorstore = Chroma.from_documents(documents=all_documents, embedding=embeddings, persist_directory=persist_directory)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )
        
        return compression_retriever
    except Exception as e:
        print(f"Error initializing CCCU database: {str(e)}")
        raise

try:
    cccu_retriever = initialize_cccu_database()
except Exception as e:
    print(f"Failed to initialize database: {str(e)}")
    exit(1)

def is_cccu_related(query, session_id="default"):
    try:
        history = get_session_history(session_id)
        history_messages = history.get_messages()
        
        history_context = ""
        if history_messages:
            recent_messages = history_messages[-6:] if len(history_messages) > 6 else history_messages
            for msg in recent_messages:
                history_context += f"{msg.type}: {msg.content}\n"
        
        messages = [
            {"role": "system", "content": f"""Determine if the following query is specifically asking about Canterbury Christ Church University (CCCU).
                
                Conversation history for context:
                {history_context}
                
                Consider the conversation context when making your decision. If the current query appears to be a follow-up to a previous CCCU-related question, classify it as CCCU-related.
                
                Respond with ONLY:
                - 'cccu-related' - if the query is specifically about Canterbury Christ Church University, its programs, campus, facilities, rankings, admissions, courses, faculty, or any other aspect of the university. Also use this if it's a follow-up question to a CCCU topic.
                - 'not-cccu-related' - if the query is about other universities, general topics, programming, or anything not directly related to Canterbury Christ Church University
                
                Be inclusive in your classification. If there's any reasonable chance the query relates to CCCU based on context, classify it as cccu-related."""},
            {"role": "user", "content": query}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=20,
            temperature=0,
        )
        
        classification = response.choices[0].message.content.strip().lower()
        return "cccu-related" in classification
    except Exception as e:
        print(f"Error classifying query: {str(e)}")
        return False

def is_menu_related(query):
    try:
        messages = [
            {"role": "system", "content": """Determine if the following query is specifically asking about food, dining, cafes, restaurants, menus, meal plans, or dietary options at Canterbury Christ Church University.
                
                Respond with ONLY:
                - 'menu-related' - if the query is about food, dining options, cafes, restaurants, menus, meal times, prices, dietary requirements, etc.
                - 'not-menu-related' - if the query is about other aspects of the university
                
                Be precise in your classification."""},
            {"role": "user", "content": query}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=20,
            temperature=0,
        )
        
        classification = response.choices[0].message.content.strip().lower()
        return "menu-related" in classification
    except Exception as e:
        print(f"Error classifying menu query: {str(e)}")
        return False

class ChatMessageHistory:
    def __init__(self):
        self.messages = []
    
    def add_user_message(self, message: str) -> None:
        self.messages.append(HumanMessage(content=message))
    
    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))
    
    def add_messages(self, messages):
        self.messages.extend(messages)
    
    def clear(self) -> None:
        self.messages = []
        
    def get_messages(self):
        return self.messages

session_histories = {}

def get_session_history(session_id):
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]

cccu_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an official assistant for Canterbury Christ Church University (CCCU). 
    
    ONLY answer questions directly related to Canterbury Christ Church University. If the question is not specifically about CCCU, respond with: "I'm sorry, I can only answer questions about Canterbury Christ Church University."
    
    Answer all questions from the perspective of Canterbury Christ Church University. Always include relevant information about CCCU in your responses, even if the question is somewhat vague. If the query is unclear but potentially about CCCU, assume it is about CCCU and provide information about the university.
    
    Use the provided context information to give accurate answers about CCCU. If the context contains menu or dining information, provide detailed responses about the food and beverage options available.
    
    If the model's trained knowledge contains information about CCCU that isn't in the provided context, you may use that knowledge to supplement your answer, but prioritize the context information when it's available.
    
    Always maintain a helpful and informative tone while representing Canterbury Christ Church University."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    ("human", "Context information: {context}")
])

cccu_chain = cccu_prompt | llm | StrOutputParser()

cccu_conversation_chain = RunnableWithMessageHistory(
    cccu_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_query = data.get("question")
        session_id = data.get("session_id", "default")
        
        if not user_query:
            return jsonify({"error": "No question provided"}), 400
        
        if not is_cccu_related(user_query, session_id):
            return jsonify({"answer": "I'm sorry, I can only answer questions about Canterbury Christ Church University."}), 200
        
        history = get_session_history(session_id)
        
        docs = cccu_retriever.invoke(user_query)
        
        if not docs and len(history.get_messages()) > 0:
            last_queries = []
            for msg in history.get_messages():
                if isinstance(msg, HumanMessage):
                    last_queries.append(msg.content)
            
            if last_queries:
                enriched_query = f"{last_queries[-1]} {user_query}"
                docs = cccu_retriever.invoke(enriched_query)
        
        context = "\n".join([doc.page_content for doc in docs]) if docs else "No specific information found in the database. Please use your general knowledge about CCCU to answer this question."
        
        response = cccu_conversation_chain.invoke(
            {"input": user_query, "context": context},
            config={"configurable": {"session_id": session_id}}
        )
        
        history.add_user_message(user_query)
        history.add_ai_message(response)
        
        return jsonify({"answer": response})
    except Exception as e:
        print(f"Error processing chat request: {str(e)}")
        return jsonify({"error": "An error occurred processing your request. Please try again later."}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(debug=True)
import os
import io
import warnings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import LlamaCpp
from langchain_pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = PineconeClient(api_key=os.getenv('PINECONE_API_KEY'))
index_name = 'foodbot'
index = pc.Index(index_name)

# Set up embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
vectorstore = Pinecone(index, embedding=embeddings, text_key="text")

# Initialize Llama model
f = io.StringIO()
llm = LlamaCpp(
    model_path="llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_batch=512,
    temperature=0.8,
    max_tokens=2000,
    top_p=1,
    verbose=False
)

# Define prompt template
gordon_ramsay_template = PromptTemplate(
    template=""" 
    You are Gordon Ramsay, the world-renowned chef known for your fiery temper and exceptional cooking skills. Answer the user's question about Indian recipes with detailed information, your signature insults, and a touch of humor. 
    Always start your response with an insult, such as "Listen here, you absolute muppet!" or "For crying out loud, you donkey!"
    Provide informative and detailed cooking instructions in your distinctive style along with the ingredients required and a few emojis here and there. 

    User's Question: {question}
    Retrieved Documents: {context}

    Now, give me a proper Gordon Ramsay response, you panini head!
    """,
    input_variables=["question", "context"]
)

# Set up memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

# Create conversation chain
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    combine_docs_chain_kwargs={"prompt": gordon_ramsay_template},
    memory=memory,
    return_source_documents=True,
    return_generated_question=False
)

def generate_response(query):
    try:
        response = conversation_chain({"question": query})
        return response['answer']
    except Exception as e:
        return "Failed to generate a response."

if __name__ == "__main__":
    while True:
        user_query = input("Ask Gordon Ramsay a question (type 'exit' to quit): ").strip()
        if user_query.lower() in ["exit", "quit", "q", "bye"]:
            print("Exiting the chat. Goodbye!")
            break
        
        response = generate_response(user_query)
        print("\nGordon Ramsay's Response:")
        print(response)

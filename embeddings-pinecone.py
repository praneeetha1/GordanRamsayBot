import os
import pandas as pd
from tqdm import tqdm
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv


load_dotenv()

pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENVIRONMENT')

pc = Pinecone(api_key=pinecone_api_key)

index_name = 'foodbot'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # number of dimension created with the miniml model
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1' 
        )
    )


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

csv_file = 'indian_food_recipes/IndianFoodDatasetCSV.csv'
df = pd.read_csv(csv_file)


def create_document(row):
    documents = []
    for col in df.columns:
        if pd.notna(row[col]):
            text = f"{col}: {row[col]}"
            documents.append(Document(page_content=text, metadata={"RecipeName": row['RecipeName'], "Column": col}))
    return documents

documents = []
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows", unit="row"):
    docs = create_document(row)
    documents.extend(docs)

print(f"Total documents created: {len(documents)}")


batch_size = 100  
for i in tqdm(range(0, len(documents), batch_size), desc="Storing documents in Pinecone", unit="batch"):
    batch_docs = documents[i:i+batch_size]
    vectorstore.add_documents(batch_docs)

print("Documents have been processed and stored in Pinecone.")


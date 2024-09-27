# Gordon Ramsay Chatbot

This is a conversational AI chatbot modeled to imitate Gordon Ramsay's cooking advice style. The chatbot uses a **ConversationalRetrievalChain** from LangChain and integrates **LlamaCpp** for language generation. The responses are delivered with humor and a touch of Gordon Ramsay’s signature insults.

## Features
- Retrieves relevant information about Indian recipes using Pinecone vectorstore.
- Uses a pre-trained model from HuggingFace for embeddings.
- Generates responses using the Llama-2 model with Gordon Ramsay’s style.
  
## Requirements
- Python 3.8+
- Pinecone API Key
- LangChain
- LangChain Community LlamaCpp
- Sentence Transformers (HuggingFace)
- dotenv
- PineconeClient

## Installation

1. Clone the repository:

```bash
git clone https://github.com/praneeetha1/GordanRamsayBot.git

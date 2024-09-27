# Gordon Ramsay Chatbot 👨🏻‍🍳

Ever hit a cooking block while making Indian food and wished for some mom-like guidance? Look no further :) – Gordon Ramsay Bot is here! This chatbot will help you cook Indian dishes with Ramsay's flair, mixing practical tips with his classic insults to spice up your experience! 

## Features
- Retrieves relevant information about Indian recipes using Pinecone vectorstore.
- Uses a pre-trained model from HuggingFace for embeddings.
- Generates responses using the Llama-2 model with Gordon Ramsay’s style.
  
## Requirements
- Python 3.8+: Make sure you're using a compatible version of Python.
- Pinecone API Key: For accessing the Pinecone vectorstore.
- LangChain: For building the conversational framework.
- LangChain Community LlamaCpp: To utilize the Llama-2 model for generating responses.
- HuggingFace Sentence Transformers: For generating embeddings from the text.
- python-dotenv: To manage environment variables from a .env file.
- Pinecone Client: To interact with the Pinecone service.

## Dataset
Here's the link to the Kaggle Dataset I've used: [Indian Food Recipes](https://www.kaggle.com/datasets/sooryaprakash12/cleaned-indian-recipes-dataset)

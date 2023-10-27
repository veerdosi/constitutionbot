# import the modules
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import os
# load .env file
from dotenv import load_dotenv
load_dotenv()

reader = PdfReader('constitution.pdf')

raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()

import pickle
with open("foo.pkl", 'wb') as f:
    pickle.dump(embeddings, f)

docsearch = FAISS.from_texts(texts, new_docsearch)
query = "what are the laws for murder"
docs = docsearch.similarity_search(query)
print(docs[0].page_content)
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
chain.run(input_documents=docs, question=query)

import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from dotenv import load_dotenv

load_dotenv()

def get_loader(filename):
    file_extension = os.path.splitext(filename)[1]
    if file_extension == ".pdf":
        loader = PyPDFLoader(filename)
    elif file_extension == ".txt":
        loader = TextLoader(filename)
    else:
        raise Exception("Unknown file type. Please provide a pdf or txt file!")
    return loader


filename = 'data/22-1534-2023-12-12.txt'
loader = get_loader(filename)
doc = loader.load()
print(doc[0].page_content)
# remove the unnecessary pages
# doc = doc[2:46]

# INDEXING
# text split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(doc)

# db creation
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
db = Chroma.from_documents(documents=split_docs, embedding=embedding)

# Retrieval
question = "When should I not use transfer learning?"
print(db.similarity_search(query=question, k=1))

# Generation
qa_template = """
You are a helpful assistant tasked with answering the question only using the provided context: {context}. Do not
answer with any prior knowledge. If the question cannot be answered using the information provided, 
answer with "I don't know".
Question: {question}
Answer:
"""
prompt = PromptTemplate(input_variables=["context", "question"],
                        template=qa_template)
llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(search_kwargs={"k": 1}), prompt=prompt)
print(chain.invoke({"context": db.as_retriever(), "query": question})["result"])

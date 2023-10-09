# import streamlit as st
import pytesseract
import glob
import os
from PIL import Image
# from io import BytesIO
# import tempfile
# import subprocess
# import os
# import base64
# import requests
# import textwrap
import unstructured_pytesseract
from langchain import HuggingFacePipeline
from langchain.document_loaders import UnstructuredPDFLoader
# from langchain.document_loaders import PyPDFLoader
# from langchain.document_loaders import TextLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.chains.question_answering import load_qa_chain
# from langchain import HuggingFaceHub
from langchain import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# Set Tesseract OCR path
unstructured_pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def process_image():
    image_files=[]
    folder_path=r'idp_app\static\in_images'
    file_extensions = ["*.pdf", "*.png", "*.jpg"]
    for extension in file_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))

    # print(image_files)
    loader = UnstructuredImageLoader(image_files[0])
    data = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    # print('data:::',data)
    docs = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings()
    global db
    db = FAISS.from_documents(docs, embeddings)
    print("PROCESSED FILES::::::::::::::::::::::")

def process_pdf():
        pdf_files=[]
        folder_path=r'idp_app\static\in_pdf'
        file_extensions = ["*.pdf", "*.png", "*.jpg"]
        for extension in file_extensions:
             pdf_files.extend(glob.glob(os.path.join(folder_path, extension)))
        loader =UnstructuredPDFLoader(pdf_files[0])
        documents = loader.load()
        # Split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=450, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        # print(docs)
        embeddings = HuggingFaceEmbeddings()
        global db
        db = FAISS.from_documents(docs, embeddings)
        print("PROCESSED FILES::::::::::::::::::::::")


def flan_main(user_input):
    llm = HuggingFacePipeline.from_model_id(model_id="declare-lab/flan-alpaca-large", task="text2text-generation", model_kwargs={"temperature":0, "max_length":1000})
    print("llm loaded")
    chain = load_qa_chain(llm, chain_type="stuff")
    print("chain loaded")
    docs = db.similarity_search(user_input)
    print("docs lodaded")
    response = chain.run(input_documents=docs, question=user_input,raw_response=True)
    return response

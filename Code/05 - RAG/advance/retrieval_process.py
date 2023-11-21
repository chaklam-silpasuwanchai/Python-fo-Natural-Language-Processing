# !pip install faiss-gpu # For CUDA 7.5+ Suppoarted GPU's.
# # OR
# !pip install faiss-cpu # For CPU Installation
# !pip install pypdf
# !pip install bs4
# from typing import list
# from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import WebBaseLoader

import faiss
import torch
import pickle
from tqdm.auto import tqdm
import langchain
import os

# from src.log_init import logger

from langchain.storage import (
    InMemoryStore,
    LocalFileStore,
    RedisStore,
    UpstashRedisStore,
)
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
import torch

fs = LocalFileStore("./cache/")

def build_embedding_model():
    embedding_model = HuggingFaceInstructEmbeddings(
            model_name='models/instructor-base',              
            model_kwargs = {
                'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            }
        )

    embedding_model = CacheBackedEmbeddings.from_bytes_store(
            embedding_model, fs, namespace=embedding_model.model_name
        )

    return embedding_model

# Release GPU memory
torch.cuda.empty_cache()

# Specify the folder path
vector_path = 'vectorstore/'
# Create the folder if it doesn't exist
if not os.path.exists(vector_path):
    print('create vector path')
    os.makedirs(vector_path)
        
def load_faiss_index(
    db_file_name : str, 
    embedding_model: HuggingFaceInstructEmbeddings
) -> langchain.vectorstores.faiss.FAISS:
    
    vector_database = FAISS.load_local(
        folder_path = db_file_name,
        embeddings  = embedding_model
    ) 
    return vector_database
    
def read_url_list(file_path):
    url_list = []
    try:
        with open(file_path, 'r') as file:
            url_list = file.read().split(',')
            # for line in file:
            #     # Remove leading and trailing whitespace, and append to the list
            #     url_list.append(line.strip())
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    return url_list


def process_db_vector(
    db_file_name : str,
    pdfs : str = None, # List of file paths to PDF documents.
    urls : list[str] = None, # text of URLs from which to retrieve documents.
    txts : list = None # List of file paths to text documents.
):    

    #### 1. Loading Documents
    ##1.1 Checking documents
    print('1/5 : Checking Data...')
    documents = []
    if pdfs:
        pdf_list = []
        for filename in os.listdir(pdfs):
            if filename.endswith('.pdf'):
                pdf_list.append(pdfs + filename)
        # logger.debug(pdf_list)

        pdf_loaders = [PyPDFLoader(pdf) for pdf in pdf_list]
        for loader in pdf_loaders:
            documents.extend(loader.load())

    if txts:
        txt_loaders = [TextLoader(txt) for txt in txts]
        for loader in txt_loaders:
            documents.extend(loader.load())
    
    if urls:
        url_list = read_url_list(urls)
        url_list = [url for url in url_list if url != '']
        # logger.debug("=========URL LIST=========")
        # logger.debug(url_list)
        # Example url_list

        # logger.debug('url check')
        loader = WebBaseLoader(url_list)
        url_documents = loader.load()
        documents.extend(url_documents)

        # logger.debug(documents)
        
    print(f'Document len : {len(documents)}')
    # logger.debug(f'Document len : {len(documents)}')

    ##1.2. Spliting
    print('2/5 : Chucking Data...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 700,
        chunk_overlap = 100
    )
    docs = text_splitter.split_documents(documents) #313

    #### 2. Embedding Model
    print('3/5 : Load embedding Model...')

    #### 3. Vector Store
    print('4/5 : Storing Vector...')
    db = FAISS.from_documents(
        documents = docs, 
        embedding = build_embedding_model())
    
    #### 4. Save vector store
    print('5/5 : Saving...')
    location_path = os.path.join(vector_path, db_file_name)
    # db.save_local(location_path)
    #### If there are db_file, skip 3 and 4
    
    # 5. Load vector Store
    # new_db = load_faiss_index(
    #     db_file_name, 
    #     embeddings) 

    print('Done...')
    return db, location_path


if __name__ == '__main__':
    folder_path = '../dataset/pdf/'
    pdf_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_list.append(folder_path + filename)

    # In case, user would like to store txt file
    # folder_path = '../docs/txt/'
    # txt_list = []
    # for filename in os.listdir(folder_path):
    #     if filename.endswith('.txt'):
    #         txt_list.append(folder_path + filename)
    file_txt = '../dataset/url_list.txt'

    db, location_path = process_db_vector(
        db_file_name = 'admin',
        pdfs = pdf_list,
        urls = file_txt,
        # txts = txt_list
    )

    
    
    


    
    
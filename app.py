from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from src.prompt import *
import os 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate  

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')   

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY   

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

prompt = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)   


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # fills the {context} for you
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)   


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = qa_chain.invoke(msg)
    print("Response : ", response["result"])
    return str(response["result"])     



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)  
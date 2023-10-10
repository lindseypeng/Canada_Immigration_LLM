import os
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


persist_directory = '/Users/linqianpeng/Documents/immigration/docs/chroma'
openai_api_key = os.environ.get('OPENAI_API_KEY')

embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

qa =RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectordb.as_retriever())

from langchain.prompts import PromptTemplate

prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  Instruction: 
  You are a immigration lawyer and your job is to answer questions on common law permanent resident (PR) application in Canada. 
  Use only information in the following paragraphs to answer the question at the end. \
  Explain the answer with reference to these paragraphs. If you don't know, say that you do not know.

  {context}
 
  Question: {question}

  Response:
  """
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}


llm = ChatOpenAI(model_name = 'gpt-3.5-turbo',temperature=0)

#qa = RetrievalQA.from_chain_type(
    #llm=OpenAI(), chain_type="stuff", retriever=vectordb.as_retriever(), chain_type_kwargs=chain_type_kwargs)

#qa = RetrievalQA.from_chain_type(
   # llm, chain_type="stuff", return_source_documents=True,retriever=vectordb.as_retriever(), chain_type_kwargs=chain_type_kwargs)


question = "how long does it take to process my PR application?"

#print(qa.run(question))

qa_chain_mr = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectordb.as_retriever(),return_source_documents=True,
    chain_type="refine"
)
result = qa_chain_mr({"query": question})
print(result)
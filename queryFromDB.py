import os
import getpass
from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import Qdrant

# from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

load_dotenv("./.env")

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

modelName = "gpt-4o"
collectionName = "whisper"
qdrantApiKey = os.getenv("QDRANT_API_KEY")
openAIKey = os.environ["OPENAI_API_KEY"]
url = os.getenv("QDRANT_HOST")

llm = ChatOpenAI(
    api_key=openAIKey,
    model=modelName,
    temperature=0
)

embeddings = OpenAIEmbeddings(
    api_key=openAIKey,
    model="text-embedding-3-small"
)

template="""Utilize as transcrições abaixo para responder à pergunta em formato de bullet points e
de forma resumida. Se não souber a resposta, diga apenas que não sabe, não tente inventar ou gerar uma resposta

    {context}

    Question: {query}
    Resposta resumida em bullet points:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["query", "context"]
)

query = "Resuma word embeddings"

qdrant = QdrantVectorStore.from_existing_collection(
    # #     documents=docs,
    embedding=embeddings,
    url=url,
    api_key=qdrantApiKey,
    collection_name=collectionName
)

docs = qdrant.similarity_search(query, k=3)

context = ""

for i in range(3):
    context += docs[i].page_content

res = LLMChain(prompt=prompt, llm=llm)

result = res.run(query=query, context=context)

print(result)

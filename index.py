import os
import getpass
import yt_dlp
import whisper
from groq import Groq
from dotenv import load_dotenv
# from openai import OpenAI

from langchain_community.llms import OpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain

# TEXT SPLITTER
from langchain.text_splitter import RecursiveCharacterTextSplitter

# DOCUMENT
from langchain.docstore.document import Document
import textwrap

load_dotenv("./.env")

def download_mp4_from_youtube(url):
    filename = "infoslack.mp4"

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "outtmpl": filename,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)

    return result

def download_audio(url):
    filename = "infoslack-audio"
    ydl_opts = {
        "format": "bestaudio",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
        "outtmpl": filename,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)

    return result

def transcriptUsingHardware():
    audio_filename = "infoslack-audio.mp3"
    model = whisper.load_model("base") # tiny | base | small | medium | large
    result = model.transcribe(
        audio=audio_filename,
    )

    # print(result["text"])
    return result

def transcriptUsingOpenAI():
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    audio_filename = "infoslack-audio.mp3"
    audio_file = open(audio_filename, "rb")

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )
    
    # print(transcript)
    return transcript

def transcriptUsingGroq():
    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
        
    client = Groq(
        # This is the default and can be omitted
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    
    audio_filename = "infoslack-audio.mp3"
    audio_file = open(audio_filename, "rb")

    result = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=audio_file,
        response_format="text"
    )
    
    return result

def writeToTxtFile(fileName, text):
    with open(fileName, "w", encoding="utf-8") as file:
        file.write(text)
        
def readTxtFile(fileName):
    with open(fileName, "r", encoding="utf-8") as f:
        text = f.read()
        
    return text

def resumeText(docs: list[Document]):
    chatgtpChatModel = "gpt-4o"
    
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=chatgtpChatModel,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    
    promptTemplate = """Escreva um resumo conciso com bullet points do texto abaixo:
    
    {text}

    RESUMO CONCISO:"""
    
    bulletPointPrompt = PromptTemplate(
        template=promptTemplate,
        input_variables=["text"]
    )

    chain = load_summarize_chain(
        llm=llm,
        chain_type="stuff",
        prompt=bulletPointPrompt
    )
    
    outputSummary = chain.run(docs)
    
    return outputSummary

def RAG(docs):
    collectionName = "whisper"
    
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )
    
    url = os.getenv("QDRANT_HOST")
    qdrantApiKey = os.getenv("QDRANT_API_KEY")
    
    # SAVE AND GET DOCUMENTS
    qdrant = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=url,
        api_key=qdrantApiKey,
        collection_name=collectionName
    )
    
    querySearch = "O que são word embeddings?"
    result = qdrant.similarity_search(querySearch, k=3)
    
    return result

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=qEmRZPUdEiw"

    download_mp4_from_youtube(url)

    download_audio(url)

    ### USING LOCAL HARDWARE
    transcriptResult = transcriptUsingHardware()

    ### USING OPENAI API
    # transcriptResult = transcriptUsingOpenAI()
        
    ## USING GROQ -- BETTER PERFORMANCE
    # transcriptResult = transcriptUsingGroq()
    
    textTranscripted = transcriptResult["text"]
    
    ### SAVE TRANSCRIPTION IN A TXT FILE
    writeToTxtFile("text.txt", textTranscripted)

    ### OPEN A TXT FILE AND READ ITS CONTENT
    textReaded = readTxtFile("text.txt")

    ### SPLIT TEXT INTO A LIST OF DOCUMENTS
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        separators=[" ", ",", "\n"]
    )

    texts = textSplitter.split_text(textReaded)
    docs = [Document(page_content=t) for t in texts]

    ### MAKE A RESUME USING GPT OPENAI
    resumeOutput = resumeText(docs)
    
    ### SAVE ON QDRANT     
    result = RAG(docs)
    
    finalText = "\n".join(doc.page_content for doc in result)
    print("finalText: ", finalText)
    
    wrappedText = textwrap.fill(
        finalText,
        width=1000,
        break_long_words=False,
        replace_whitespace=False
    )

    writeToTxtFile("wrappedText.txt", wrappedText)
    print(wrappedText)

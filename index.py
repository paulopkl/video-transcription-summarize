import os
import yt_dlp
import whisper
from groq import Groq
from dotenv import load_dotenv
from openai import OpenAI

from langchain import OpenAI, LLMChain
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

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

if __name__ == "__main__":
    # url = "https://www.youtube.com/watch?v=qEmRZPUdEiw"

    # download_mp4_from_youtube(url)

    # download_audio(url)

    ### USING LOCAL HARDWARE
    audio_filename = "infoslack-audio.mp3"
    model = whisper.load_model("base") # tiny | base | small | medium | large
    result = model.transcribe(audio_filename)

    print(result["text"])

    ### USING OPENAI API
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # audio_filename = "infoslack-audio.mp3"
    audio_file = open(audio_filename, "rb")

    # transcript = client.audio.transcriptions.create(
    #     model="whisper-1",
    #     file=audio_file,
    #     response_format="text"
    # )

    # print(transcript)

    with open("text.txt", "w") as file:
        file.write(result)
    
    ## USING GROQ -- BETTER PERFORMANCE
    client = Groq(
        # This is the default and can be omitted
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    resp = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=audio_file,
        response_format="text"
    )

    ### MAKE A RESUME
    

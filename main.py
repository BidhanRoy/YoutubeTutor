# Import necessary modules
from typing import List
import streamlit as st
from streamlit_chat import message

# Import langchain modules
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

# Import llama_index modules
from llama_index import download_loader, GPTVectorStoreIndex
from llama_index.readers.schema.base import Document

# Import YouTube Transcript API
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(youtube_link: str) -> str:
    """Extracts the video ID from a YouTube link."""
    return youtube_link.split("?v=")[-1]

def get_transcript(video_id: str) -> List[dict]:
    """Fetches the transcript of a YouTube video given its ID."""
    return YouTubeTranscriptApi.get_transcript(video_id)

def chunk_transcript(transcript: List[dict], chunk_size: int = 1000) -> List[Document]:
    """
    Splits the transcript into chunks of a specified size.
    Returns a list of Document objects.
    """
    results = []
    current_chunk = ""
    for chunk in transcript:
        if len(current_chunk) + len(chunk["text"]) <= chunk_size:
            current_chunk += chunk["text"]
        else:
            results.append(Document(current_chunk))
            current_chunk = chunk["text"]
    results.append(Document(current_chunk))
    return results

# Initialize the question generator
question_generator = LLMChain(
    llm=OpenAI(temperature=0),
    prompt=PromptTemplate.from_template(
        """
        As a YoutubeTutor, your role involves teaching and assessing a student's ability to understand a specific subject matter. Your current task is to examine a transcript of a Youtube video and create a question that tests a student's level of attentiveness to the video's content. Your aim is to formulate a single, highly specific question that can be answered with just one word. It is essential that you base your inquiry on the provided context: {context}. Multiple-word answers are not acceptable. Please ensure that your question is the most significant one possible.
        """
    )
)

# Initialize the answer evaluator
answer_evaluator = LLMChain(
    llm=OpenAI(temperature=0),
    prompt=PromptTemplate.from_template(
        """
        As a YouTube tutor, you have been presented with the context: {context} and asked the question: {question}.
        The user has provided the answer: {answer}. Your task is to determine whether the answer given is valid within the context.
        Your response should be either 'correct' or 'wrong'
        """
    )
)

CURRENT_INDEX = "current_index"
YOUTUBE_LINK = "youtube_link"
DOCUMENTS = "documents"
PAYLOAD = "payload"

def display_next_question():
    current_index = st.session_state[CURRENT_INDEX]
    context = st.session_state[DOCUMENTS][current_index].text
    question = question_generator(inputs={"context": context}, return_only_outputs=True)['text']

    print("\n\nCurrent index = {}\nContext = {}\nQuestion = {}\n\n".format(
        current_index, context, question
    ))

    message(question)
    user_answer = st.text_input("Type your answer: ", key = str(current_index) + '_answer')
    if st.button('Submit Answer', key = str(current_index) + '_submit'):
        handle_answer(context, user_answer, question)

def handle_answer(context, user_answer, question):
    print('inside handle answer....')
    result = answer_evaluator(
        inputs={
            "context": context,
            "answer": user_answer,
            "question": question
        },
        return_only_outputs=True
    )['text']

    current_index = st.session_state[CURRENT_INDEX]
    
    st.session_state[PAYLOAD][current_index] = {}
    st.session_state[PAYLOAD][current_index]["question"] = question
    st.session_state[PAYLOAD][current_index]["answer"] = user_answer
    st.session_state[PAYLOAD][current_index]["result"] = result

    message(user_answer, is_user=True, key=str(current_index) + "_user")
    message(result, key = str(current_index) + "_result")

    print("current index = {}".format(current_index))
    print("payload = {}".format(str(st.session_state[PAYLOAD][current_index])))

    print('incrementing index...')
    st.session_state[CURRENT_INDEX] += 1

    display_next_question()

def main():
    if CURRENT_INDEX not in st.session_state:
        print("\n....initiating current index....\n")
        st.session_state[CURRENT_INDEX] = 0

    if YOUTUBE_LINK not in st.session_state:
        st.session_state[YOUTUBE_LINK] = None

    if DOCUMENTS not in st.session_state:
        st.session_state[DOCUMENTS] = []

    if PAYLOAD not in st.session_state:
        st.session_state[PAYLOAD] = {}

    st.title("YouTube Tutor")

    if not st.session_state[YOUTUBE_LINK]:
        st.session_state[YOUTUBE_LINK] = st.text_input("Enter YouTube video link", key="youtube")

    if not st.session_state[YOUTUBE_LINK]:
        return
    
    message("Test on video: {}".format(st.session_state[YOUTUBE_LINK]))

    if not st.session_state[DOCUMENTS]:
        youtube_link = st.session_state[YOUTUBE_LINK]
        video_id = extract_video_id(youtube_link)
        transcript = get_transcript(video_id)
        documents = chunk_transcript(transcript)
        st.session_state[DOCUMENTS] = documents

    display_next_question()
    
    print("changed index = {}".format(st.session_state[CURRENT_INDEX]))

if __name__ == "__main__":
    main()

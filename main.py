from typing import List
import streamlit as st
from streamlit_chat import message
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from llama_index.readers.schema.base import Document
from youtube_transcript_api import YouTubeTranscriptApi
import textwrap

# Constants
CURRENT_INDEX = "current_index"
YOUTUBE_LINK = "youtube_link"
DOCUMENTS = "documents"

# Utility functions
def extract_video_id(youtube_link: str) -> str:
    return youtube_link.split("?v=")[-1]

def get_transcript(video_id: str) -> List[dict]:
    return YouTubeTranscriptApi.get_transcript(video_id)

def chunk_transcript(transcript: List[dict], chunk_size: int = 3000) -> List[Document]:
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

# Initialize AI models
question_prompt = textwrap.dedent("""
    As a YoutubeTutor, your role involves teaching and assessing a student's ability 
    to understand a specific subject matter. Your current task is to examine a transcript 
    of a Youtube video and create a question that tests a student's level of attentiveness 
    to the video's content. Your aim is to formulate a single, highly specific question 
    that can be answered with just one word. It is essential that you base your inquiry 
    on the provided context: {context}. Multiple-word answers are not acceptable. Please 
    ensure that your question is the most significant and technical one possible.
    """)

answer_prompt = textwrap.dedent("""
    As a YouTube tutor, you have been presented with the context: {context} and asked 
    the question: {question}. The user has provided the answer: {answer}. Your task is 
    to determine whether the answer given is valid within the context. Your response 
    should be either 'correct' or 'wrong'. Add some context on the right answer.
    """)

question_generator = LLMChain(llm=OpenAI(temperature=0), 
                               prompt=PromptTemplate.from_template(question_prompt))
answer_evaluator = LLMChain(llm=OpenAI(temperature=0), 
                             prompt=PromptTemplate.from_template(answer_prompt))

# Main app functions
def display_next_question():
    current_index = st.session_state[CURRENT_INDEX]
    if current_index >= len(st.session_state[DOCUMENTS]):
        return

    context = st.session_state[DOCUMENTS][current_index].text
    question = question_generator(inputs={"context": context}, return_only_outputs=True)['text'].strip()
    message(question, key=str(current_index) + '_question', avatar_style="big-smile")
    user_answer = st.text_input("Type your answer: ", key = str(current_index) + '_answer')
    if st.button('Submit Answer', key = str(current_index) + '_submit'):
        handle_answer(context, user_answer, question)

def handle_answer(context, user_answer, question):
    result = answer_evaluator(inputs={"context": context, 
                                       "answer": user_answer, 
                                       "question": question}, 
                              return_only_outputs=True)['text'].strip();

    current_index = st.session_state[CURRENT_INDEX]
    message(user_answer, is_user=True, key=str(current_index) + "_user")
    message(result, key = str(current_index) + "_result", avatar_style="big-smile")
    st.session_state[CURRENT_INDEX] += 1
    display_next_question()

def main():
    st.session_state.setdefault(CURRENT_INDEX, 0)
    st.session_state.setdefault(YOUTUBE_LINK, None)
    st.session_state.setdefault(DOCUMENTS, [])
    st.title("YouTube Tutor")

    if not st.session_state[YOUTUBE_LINK]:
        st.session_state[YOUTUBE_LINK] = st.text_input("Enter YouTube video link", key="youtube")
    if not st.session_state[YOUTUBE_LINK]:
        return

    message("Test on video: {}".format(st.session_state[YOUTUBE_LINK]), avatar_style="big-smile")

    if not st.session_state[DOCUMENTS]:
        youtube_link = st.session_state[YOUTUBE_LINK]
        video_id = extract_video_id(youtube_link)
        transcript = get_transcript(video_id)
        documents = chunk_transcript(transcript)
        st.session_state[DOCUMENTS] = documents

    display_next_question()

if __name__ == "__main__":
    main()

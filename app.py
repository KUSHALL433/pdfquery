# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate


# genai.configure(api_key="AIzaSyBcBbja2RjoY73qaWg-YpzbYTUv8FUbK1Q")





# def get_pdf_text(pdf_docs):
#     text=""
#     for pdf in pdf_docs:
#         pdf_reader= PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text+= page.extract_text()
#     return  text



# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key='AIzaSyBcBbja2RjoY73qaWg-YpzbYTUv8FUbK1Q')
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")


# def get_conversational_chain():

#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro",
#                              temperature=0.3,google_api_key='AIzaSyBcBbja2RjoY73qaWg-YpzbYTUv8FUbK1Q')

#     prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain



# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key="AIzaSyBcBbja2RjoY73qaWg-YpzbYTUv8FUbK1Q")
    
    
#     new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()

    
#     response = chain(
#         {"input_documents":docs, "question": user_question}
#         , return_only_outputs=True)

#     print(response)
#     st.write("Reply: ", response["output_text"])




# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with PDF using PDFQuery💁")

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")



# if __name__ == "__main__":
#     main()


from urllib import request
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import http.client
import json
from urllib.parse import quote

genai.configure(api_key="AIzaSyBcBbja2RjoY73qaWg-YpzbYTUv8FUbK1Q")

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key='AIzaSyBcBbja2RjoY73qaWg-YpzbYTUv8FUbK1Q')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible and diagramatically from the provided context, make sure to provide all the details,if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3,google_api_key='AIzaSyBcBbja2RjoY73qaWg-YpzbYTUv8FUbK1Q')

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key="AIzaSyBcBbja2RjoY73qaWg-YpzbYTUv8FUbK1Q")
        
        
        new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()

        
        response = chain(
            {"input_documents":docs, "question": user_question}
            , return_only_outputs=True)

        print(response)
        st.write("Reply: ", response["output_text"])
        return user_question
    except Exception as e:
        st.error(f"Error with PDF interaction: {e}")
        return None


def search_articles(query):
    conn = http.client.HTTPSConnection("google-search72.p.rapidapi.com")
    
    headers = {
        'x-rapidapi-key': "14f44c3385msh3b4f3febee40bccp14b5e0jsna88e85c14032", 
        'x-rapidapi-host': "google-search72.p.rapidapi.com"
    }

    query_encoded = quote(query)

    try:
        conn.request("GET", f"/search?q={query_encoded}&lr=en-US&num=10", headers=headers)
        res = conn.getresponse()

        data = res.read()
        articles = json.loads(data.decode("utf-8"))
        return articles.get("items", [])
    except Exception as e:
        st.error(f"Error fetching articles: {e}")
        return []


def display_articles(query):
    if query:
        st.subheader(f"Articles 📝 Related to: {query}")
        articles = search_articles(query)
        
        if articles:
            for article in articles:
                title = article.get("title", "No Title Available")
                link = article.get("link", "#")
                image_url = article.get("thumbnail", "")
                st.subheader(title)
                st.write(f"[Read More]({link})")

                if image_url:  # Display thumbnail if available
                    st.image(image_url, use_column_width=True)
        else:
            st.write("No articles found for this query.")

    

def get_youtube_videos(query):
    try:
        conn = http.client.HTTPSConnection("youtube-v31.p.rapidapi.com")
        
        headers = {
            'x-rapidapi-key': "14f44c3385msh3b4f3febee40bccp14b5e0jsna88e85c14032",
            'x-rapidapi-host': "youtube-v31.p.rapidapi.com"
        }
        
        query_encoded = query.replace(" ", "%20")
        
        conn.request("GET", f"/search?part=id%2Csnippet&q={query_encoded}&type=video&maxResults=5", headers=headers)
        res = conn.getresponse()
        data = res.read()
        videos = json.loads(data.decode("utf-8"))["items"]
        return videos
    except Exception as e:
        st.error(f"Error fetching YouTube videos: {e}")
        return []

def display_youtube_videos(query):
    if query:
        st.subheader(f"YouTube Videos 🎥 Related to: {query}")
        videos = get_youtube_videos(query)
        
        if videos:
            for video in videos:
                title = video["snippet"]["title"]
                video_url = f"https://www.youtube.com/watch?v={video['id']['videoId']}"
                thumbnail_url = video["snippet"]["thumbnails"]["high"]["url"]
                st.markdown(f"[{title}]({video_url})")
                
                # Hyperlink the thumbnail
                st.image(thumbnail_url, use_column_width=True, caption=f"[Watch Video]({video_url})")
        else:
            st.write("No YouTube videos found for this query.")



def main():
    st.set_page_config("Chat PDF + YouTube Videos")
    st.header("Chat with PDF using PDFQuery 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        query = user_input(user_question)
        if query:  
            display_youtube_videos(query)
            display_articles(query)
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing Complete! You can now ask questions.")
                    
            else:
                st.error("Please upload at least one PDF file.")
    
    

if __name__ == "__main__":
    main()




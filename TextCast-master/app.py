import os
import sys
import logging
import subprocess
from dotenv import load_dotenv

# Dependency checks
try:
    import flask
    from flask import Flask, render_template, jsonify, request
    import whisper
    from gtts import gTTS
    import pdfminer
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextBoxHorizontal
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_groq import ChatGroq
    from langchain.prompts import ChatPromptTemplate
    from groq import Groq
    from langchain_community.document_loaders.text import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required dependencies using:")
    print("pip install flask whisper gtts pdfminer.six langchain-groq groq langchain-huggingface")
    sys.exit(1)

# Check FFmpeg
try:
    subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
except (subprocess.CalledProcessError, FileNotFoundError):
    print("FFmpeg is not installed or not in PATH. Please install FFmpeg.")
    sys.exit(1)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 500 MB max file size

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAIN_DIR = "Research Paper"    # Stores uploaded papers
INPUT_DIR = "Input_audio"   # Stores user input audio
PODCAST_PATH = os.path.join("static", "audio")      # Stores podcast audio
SUMMARIES_FILE = os.path.join(MAIN_DIR, "summaries.txt")        # Stores each page summary
os.makedirs(MAIN_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(PODCAST_PATH, exist_ok=True)

# Global models and retriever
try:
    WHISPER_MODEL = whisper.load_model("base")
except Exception as e:
    logger.warning(f"Whisper model can't be loaded, {e}")

try:
    EMBEDDINGS = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
except Exception as e:
    logger.warning(f"Embedding is not able to load {e}")
    sys.exit(1)

load_dotenv()
try:
    api = os.environ["GROQ_API_KEY"]
    LLM = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api)
except Exception as e:
    logger.warning(f"LLM is not able to load {e}")
    sys.exit()

FAISS_RETRIEVER = None

def extract_page_text(pdf_path):
    """Extract text from each page of the PDF."""
    try:
        pages = []
        for page_layout in extract_pages(pdf_path):
            page_text = "".join(
                element.get_text() for element in page_layout
                if isinstance(element, LTTextBoxHorizontal)
            ).strip()
            pages.append(page_text)
        return pages
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
        raise ValueError(f"PDF extraction failed: {str(e)}")

def summarize_page(page_content):
    """Generate a 200-300 word summary for every single page."""
    try:
        template = """Summarize the following research paper page in 200-300 words, focusing on its key insights, methods, and important discussions in a clear, simple tone for a general audience. Avoid technical jargon—explain complex ideas in everyday terms. If the page lacks specific details, infer the main takeaway based on the context. Here's the page content:

{page_content}

Provide the summary in this format:
- Key Insight: [One sentence on the page's main takeaway]
- Methods or Approach:
- Important Discussion: """
        prompt = ChatPromptTemplate.from_template(template)
        chain = {'page_content': RunnablePassthrough()} | prompt | LLM | StrOutputParser()
        return chain.batch(page_content, {"max_concurrency": 1})
    except Exception as e:
        logger.error(f"Failed to summarize pages: {e}")
        raise ValueError(f"Summarization failed: {str(e)}")

PODCAST_TEMPLATE = ChatPromptTemplate.from_template(
    """You are an AI podcast host named Sophia, and your guest, Mia, is an expert in the domain of the provided research paper.

Create a 750-word podcast script (approximately 5-6 minutes when spoken) where:
1. Sophia starts with an enthusiastic, casual intro that hooks listeners into the topic.
2. Sophia and Mia discuss the research paper's key insights in simple, relatable terms, bouncing ideas off each other naturally.
3. They explore the real-world impact of the findings together, sharing excitement and curiosity.
4. Sophia ends with a fun, thought-provoking closing question, and Mia responds energetically.

Use a lively, conversational tone with natural dialogue only between Sophia and Mia. Format the script with each speaker's lines prefixed as:
Sophia:
Mia:

Include paralanguage like laughs, pauses, excitedly, or thoughtfully to make it engaging, but don't mention these words in the spoken text. Let them interrupt, agree, or riff off each other for a real discussion feel. Research Paper content: {context}

Now generate the podcast script."""
)

def get_response(content):
    """Generate podcast script from summarized content."""
    try:
        system_prompt = PODCAST_TEMPLATE.format(context=content)
        chat_completion = LLM.invoke([{"role": "system", "content": system_prompt}])
        return chat_completion.content
    except Exception as e:
        logger.error(f"Failed to generate podcast script: {e}")
        raise ValueError(f"Script generation failed: {str(e)}")

def script_to_audio(script, output_path):
    """Convert script to audio and combine into a single file."""
    try:
        # Ensure script is a string if it's not already
        if not isinstance(script, str):
            script = str(script)

        sophia_lines = [line.replace("Sophia:", "").strip() for line in script.split("\n") if line.startswith("Sophia:")]
        mia_lines = [line.replace("Mia:", "").strip() for line in script.split("\n") if line.startswith("Mia:")]
        audio_files = []

        # Adjust the zip to handle cases with unequal line counts
        for i in range(max(len(sophia_lines), len(mia_lines))):
            if i < len(sophia_lines):
                tts = gTTS(sophia_lines[i], lang="en", tld="co.za")
                sophia_path = os.path.join(PODCAST_PATH, f"sophia_{i}.mp3")
                tts.save(sophia_path)
                audio_files.append(sophia_path)

            if i < len(mia_lines):
                tts = gTTS(mia_lines[i], lang="en", tld="co.uk")
                mia_path = os.path.join(PODCAST_PATH, f"mia_{i}.mp3")
                tts.save(mia_path)
                audio_files.append(mia_path)

        if audio_files:
            ffmpeg_cmd = ["ffmpeg", "-i", f"concat:{'|'.join(audio_files)}", "-acodec", "copy", output_path]
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            for temp_file in audio_files:
                os.remove(temp_file)
            return True
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr.decode()}")
        return False
    except Exception as e:
        logger.error(f"Error in script_to_audio: {e}")
        return False

def initialize_vector_store(summaries):
    """Initialize FAISS vector store from summaries."""
    try:
        with open(SUMMARIES_FILE, "w") as f:
            f.write("\n\n".join(summaries))
        docs = TextLoader(SUMMARIES_FILE).load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        return FAISS.from_documents(documents=split_docs, embedding=EMBEDDINGS).as_retriever()
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        raise ValueError(f"Vector store initialization failed: {str(e)}")

def speech_to_text(audio_path):
    """Transcribe audio to text using Whisper."""
    try:
        logger.info("Transcribing audio...")
        result = WHISPER_MODEL.transcribe(audio_path)
        question = result["text"].strip()
        logger.info(f"Transcribed: {question}")
        return question
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return None

def generate_response(question, retriever):
    """Generate a conversational response using RAG."""
    try:
        INTERACTIVE_TEMPLATE = ChatPromptTemplate.from_template(
    """You're Mia, an expert on a research paper's page summaries. A user asked about the paper. Create a short, lively response where:

Mia answers simply, using the summaries and her expertise, with natural flow.
If the question's unclear, say: "I can't quite catch that—could you repeat it?"

Use the summaries: {context}

Question: {question}

Keep it dialogue-only, short, and natural."""
        )
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | INTERACTIVE_TEMPLATE
            | LLM
            | StrOutputParser()
        )
        logger.info("Generating response...")
        response = rag_chain.invoke(question)
        return response
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return "Hey, I'm not able to understand the question. Please, repeat again!"
    
def script_to_audio_answer(script, output_path):
    """Convert script to audio and combine into a single file."""
    try:
        audio_files = []
        with open("audio_script", "w") as file:
            file.writelines(script)

        # Ensure script is a string, split by lines if needed
        if isinstance(script, str):
            script_lines = script.split('\n')
        else:
            script_lines = script

        for i, line in enumerate(script_lines):
            tts = gTTS(line, lang="en", tld="co.uk")
            mia_path = os.path.join(PODCAST_PATH, f"mia_{i}.mp3")
            tts.save(mia_path)
            audio_files.append(mia_path)

        if audio_files:
            ffmpeg_cmd = ["ffmpeg", "-i", f"concat:{'|'.join(audio_files)}", "-acodec", "copy", output_path]
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            for temp_file in audio_files:
                os.remove(temp_file)
            return True
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr.decode()}")
        return False
    except Exception as e:
        logger.error(f"Error in script_to_audio_answer: {e}")
        return False

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    global FAISS_RETRIEVER
    # Clear podcast directory at startup
    for file in os.listdir(PODCAST_PATH):
        file_path = os.path.join(PODCAST_PATH, file)
        if os.path.exists(file_path) and file.endswith(".mp3"):
            os.remove(file_path)
            logger.info(f"Removed old audio file: {file_path}")

    # Clear main directory at startup
    for file in os.listdir(MAIN_DIR):
        file_path = os.path.join(MAIN_DIR, file)
        if os.path.exists(file_path) and (file.endswith(".pdf") or file.endswith(".txt")):
            os.remove(file_path)
            logger.info(f"Removed old main file: {file_path}")

    file_path = None
    try:
        if 'pdf' not in request.files:
            logger.error("No file part in request")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['pdf']
        if file.filename == "":
            logger.error("No file selected")
            return jsonify({"error": "No file selected"}), 400

        file_path = os.path.join(MAIN_DIR, file.filename)
        file.save(file_path)
        content = extract_page_text(file_path)
        logger.info("Each page text extracted")
        summaries = summarize_page(content)
        logger.info("Page summaries generated")
        FAISS_RETRIEVER = initialize_vector_store(summaries)  # Initialize retriever once
        podcast_script = get_response(summaries)
        logger.info("Podcast script generated")
        success = script_to_audio(podcast_script, os.path.join(PODCAST_PATH, "podcast.mp3"))

        if success:
            logger.info(f"Podcast saved to {PODCAST_PATH}/podcast.mp3")
            return jsonify({"podcastUrl": "/static/audio/podcast.mp3"})
        else:
            logger.error("Podcast generation failed")
            return jsonify({"error": "Podcast generation failed"}), 500
    except ValueError as e:
        logger.error(f"Processing error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {e}")
        return jsonify({"error": "Internal server error"}), 500
    
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

@app.route("/ask-question", methods=["POST"])
def ask_question():
    audio_path = None
    try:
        if "audio" not in request.files:
            logger.error("No audio uploaded")
            return jsonify({"error": "No audio uploaded"}), 400

        audio_file = request.files["audio"]
        audio_path = os.path.join(INPUT_DIR, "question.webm")
        audio_file.save(audio_path)

        for file in os.listdir(PODCAST_PATH):
            file_path = os.path.join(PODCAST_PATH, file)
            if os.path.exists(file_path) and file.endswith(".mp3") and file != "podcast.mp3":
                os.remove(file_path)
                logger.info(f"Removed old audio file: {file_path}")

        question = speech_to_text(audio_path)
        if not question:
            raise ValueError("Failed to transcribe audio")

        if not FAISS_RETRIEVER:
            raise ValueError("Vector store not initialized. Upload a PDF first.")

        response = generate_response(question, FAISS_RETRIEVER)
        success = script_to_audio_answer(response, os.path.join(PODCAST_PATH, "response.mp3"))

        if success:
            logger.info(f"Response saved to {PODCAST_PATH}/response.mp3")
            return jsonify({"responseUrl": "/static/audio/response.mp3"})
        else:
            logger.error("Response generation failed")
            return jsonify({"error": "Response generation failed"}), 500
    except ValueError as e:
        logger.error(f"Processing error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
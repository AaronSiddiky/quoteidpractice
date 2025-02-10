from typing import List, Tuple, Dict, Union
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel
import os
import fitz  # PyMuPDF for handling PDFs
from PIL import Image
import io
from langchain.tools import StructuredTool
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_cors import cross_origin

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Check if OPENAI_API_KEY is in environment, otherwise use hardcoded key for development
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = "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"
os.environ["OPENAI_API_KEY"] = api_key

# Add this near the top of the file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class EvaluateGuessInput(BaseModel):
    guess: str
    correct_book: str

class QuoteAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.tools = self.get_tools()
        self.prompt = self.get_prompt()
        self.agent = self.setup_agent()
        self.vector_store = None
        self.books = []
        self.current_page = None
        self.used_quotes = set()  # Track used quotes in this session

    def get_tools(self):
        def select_quote(passage: str) -> str:
            """Select a meaningful quote from the passage."""
            prompt = """
            Select a meaningful quote from this passage. Return only the quote itself, 
            without any additional formatting or explanation.
            
            Passage: {passage}
            """
            return self.llm.invoke(prompt.format(passage=passage)).content

        def evaluate_guess(guess: str, correct_book: str) -> Dict[str, Union[bool, str]]:
            """Evaluate if the guess matches the book semantically and provide feedback."""
            prompt = f"""
            Evaluate if the guess '{guess}' refers to the book '{correct_book}'.
            Consider:
            1. Different ways people might refer to the book
            2. Author names or common abbreviations
            3. Partial matches or alternative titles
            """
            
            response = self.llm.invoke(prompt)
            semantic_match = "ibn arabi" in guess.lower() or "translator" in guess.lower() or "desires" in guess.lower()
            
            return {
                "is_correct": semantic_match,
                "feedback": response.content
            }

        tools = [
            Tool(
                name="select_quote",
                description="Select a meaningful quote from the passage",
                func=select_quote
            ),
            StructuredTool(
                name="evaluate_guess",
                description="Evaluate if a guess matches the correct book",
                func=evaluate_guess,
                args_schema=EvaluateGuessInput
            )
        ]
        return tools

    def get_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", "You are a professor testing students' knowledge of literature through quotes. Select meaningful passages and evaluate their guesses."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    def setup_agent(self):
        functions = [convert_to_openai_function(t) for t in self.tools]
        llm_with_tools = self.llm.bind(functions=functions)
        
        agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"])
            }
            | self.prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )
        
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def add_book(self, filepath: str, title: str, author: str, page_range: tuple = None):
        # Load PDF with better settings
        loader = PyPDFLoader(
            filepath,
            extract_images=False
        )
        pages = loader.load()
        
        print(f"Total pages loaded: {len(pages)}")
        
        # Filter pages based on page range and odd numbers (English translations)
        if page_range:
            filtered_pages = [
                page for page in pages 
                if (
                    page_range[0] <= page.metadata.get('page', 0) <= page_range[1]
                    and page.metadata.get('page', 0) % 2 == 1  # Only odd pages for English
                )
            ]
            print(f"Pages after filtering: {len(filtered_pages)}")
            print(f"Sample page numbers: {[p.metadata.get('page', 0) for p in filtered_pages[:5]]}")
            
            # Print sample content from first few pages
            print("\nSample content from first few pages:")
            for p in filtered_pages[:2]:
                print(f"\nPage {p.metadata.get('page', 0)}:")
                print(p.page_content[:200])
            
            pages = filtered_pages
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Increased chunk size
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "? ", "! "]  # Added spaces after punctuation
        )
        docs = text_splitter.split_documents(pages)
        print(f"Number of text chunks: {len(docs)}")
        
        embeddings = OpenAIEmbeddings()
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(docs, embeddings)
        else:
            self.vector_store.add_documents(docs)
            
        self.books.append({
            "title": title, 
            "author": author, 
            "filepath": filepath,
            "page_range": page_range
        })

    def get_random_quote(self):
        if not self.vector_store or not self.books:
            raise ValueError('No books loaded in the game')

        import random
        random_book = random.choice(self.books)
        
        # Get passages with more variety and randomness
        search_queries = [
            "Find a poetic verse or passage",
            "Find a meaningful quote",
            "Find an important passage",
            "Find a key verse",
            "Find a significant quote"
        ]
        
        max_attempts = 5  # Limit attempts to prevent infinite loops
        for attempt in range(max_attempts):
            query = random.choice(search_queries)
            results = self.vector_store.similarity_search(
                query,
                k=5  # Reduced to get more focused results
            )
            
            print(f"\nUsing query: {query}")
            
            valid_results = [
                r for r in results 
                if (
                    len(r.page_content.strip()) > 0
                    and not r.page_content.lower().startswith("the translator")
                    and not any(arabic_char in r.page_content for arabic_char in 'ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْ')
                    and r.page_content.strip() not in self.used_quotes
                    and len(r.page_content.split()) >= 10  # Ensure quote is substantial
                    and len(r.page_content.split()) <= 100  # But not too long
                )
            ]

            if valid_results:
                selected_result = random.choice(valid_results)
                try:
                    response = self.agent.invoke({
                        "input": f"""
                        Select a meaningful quote from this passage. Choose something poetic and memorable.
                        The quote should be self-contained and make sense on its own.
                        
                        Passage (Page {selected_result.metadata.get('page', 0)}):
                        {selected_result.page_content}
                        
                        Return your response in exactly this format:
                        QUOTE: (the quote)
                        PAGE: {selected_result.metadata.get('page', 0)}
                        """,
                        "chat_history": []
                    })

                    response_text = response["output"]
                    quote_part = response_text.split("QUOTE:")[1].split("PAGE:")[0].strip()
                    page_part = response_text.split("PAGE:")[1].strip()
                    page_number = int(page_part)
                    
                    # Verify the quote isn't too short
                    if len(quote_part.split()) < 5:
                        continue
                    
                    self.used_quotes.add(quote_part.strip())
                    
                    return {
                        "quote": quote_part,
                        "correct_book": random_book,
                        "page": page_number
                    }
                except Exception as e:
                    print(f"Error processing quote (attempt {attempt + 1}): {str(e)}")
                    continue

        raise ValueError("Could not find new unique quotes. Try with different search terms.")

    def show_page_image(self, filepath: str, page_number: int):
        """Display the page image where the quote was found."""
        try:
            pdf_document = fitz.open(filepath)
            
            # For odd pages (Arabic), show the next page (English)
            if page_number % 2 == 1:
                page_number += 1
            
            # Convert to 0-based index for PDF
            page = pdf_document[page_number - 1]
            
            # Get the page as an image with higher resolution
            zoom = 2
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            image_path = f"page_{page_number}.png"
            image.save(image_path, quality=95)
            print(f"\nPage image saved as: {image_path}")
            
            pdf_document.close()
            return image_path
        except Exception as e:
            print(f"Error showing page image: {str(e)}")
            return None

    def evaluate_guess(self, guess: str, correct_book: str):
        """Evaluate the user's guess using semantic understanding."""
        response = self.agent.invoke({
            "input": f"""
            Evaluate this guess: '{guess}'
            For the book: '{correct_book}'
            Consider variations of the title, author name, and common references.
            """,
            "chat_history": []
        })

        # Get evaluation from the tool
        tool_response = self.tools[1].func(guess, correct_book)
        
        return {
            "is_correct": tool_response["is_correct"],
            "feedback": tool_response["feedback"]
        }

# Add Flask routes
@app.route('/')
def home():
    return "Quote Game is running!"

# Modify the book initialization
agent = QuoteAgent()  # Create a single instance
book = {
    "title": "The Translator of Desires",
    "author": "Ibn 'Arabi",
    "filepath": os.path.join(BASE_DIR, "books", "translator_of_desires.pdf"),
    "page_range": (41, 282)
}
# Initialize the agent with the book
agent.add_book(**book)

@app.route('/quote', methods=['GET'])
@cross_origin()
def get_quote():
    try:
        # Check if PDF exists
        if not os.path.exists(book["filepath"]):
            return jsonify({"error": f"PDF file not found at {book['filepath']}"}), 500
            
        result = agent.get_random_quote()
        image_path = agent.show_page_image(
            result["correct_book"]["filepath"], 
            result["page"]
        )
        result["image_path"] = image_path
        return jsonify(result)
    except Exception as e:
        print(f"Error in get_quote: {str(e)}")  # Add logging
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate', methods=['POST'])
@cross_origin()
def evaluate_guess():
    try:
        data = request.json
        guess = data.get('guess')
        correct_book = data.get('correct_book')
        
        evaluation = agent.evaluate_guess(guess, correct_book)
        return jsonify(evaluation)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add new route to serve images
@app.route('/image/<path:filename>')
@cross_origin()
def serve_image(filename):
    try:
        return send_file(filename, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 404

# Modify the main block
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Changed default port to 8000
    app.run(host="0.0.0.0", port=port) 
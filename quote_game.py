import os
import random
from typing import List, Tuple
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

class QuoteGame:
    def __init__(self, books_dir: str = "books"):
        self.books_dir = books_dir
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased to accommodate full paragraphs
            chunk_overlap=0,   # No overlap needed for paragraphs
            length_function=len,
            separators=["\n\n", "\n", ". "]  # Prioritize paragraph breaks
        )
        self.passages = []
        self.book_titles = []
        self.vector_store = None
        
        # Criteria for important quotes
        self.importance_criteria = [
            "This is a pivotal moment in the story",
            "This reveals character development",
            "This is a major plot twist",
            "This shows conflict or tension",
            "This reveals a key theme of the story",
            "This is a climactic moment",
            "This changes the direction of the story",
            "This reveals important character motivation",
            "This is a defining moment for the protagonist",
            "This shows the resolution of a major conflict"
        ]
        self.importance_embeddings = self.model.encode(self.importance_criteria)
        self.load_books()
    
    def is_quote_significant(self, quote: str, threshold: float = 0.3) -> bool:
        """Determine if a quote is significant based on semantic similarity to importance criteria."""
        quote_embedding = self.model.encode([quote])[0]
        similarities = np.dot(self.importance_embeddings, quote_embedding)
        max_similarity = np.max(similarities)
        return max_similarity > threshold
    
    def get_random_quote(self) -> Tuple[str, str]:
        """Return a random significant quote and its corresponding book title."""
        if not self.passages:
            raise ValueError("No passages loaded!")
        
        # Try up to 10 times to find a significant quote
        for _ in range(10):
            random_idx = random.randint(0, len(self.passages) - 1)
            quote, book_title = self.passages[random_idx]
            
            if self.is_quote_significant(quote):
                return quote, book_title
        
        # If no significant quote found after 10 attempts, return the last tried quote
        return quote, book_title
    
    def load_books(self):
        """Load all PDF books from the books directory and create vector store."""
        for filename in os.listdir(self.books_dir):
            if filename.endswith('.pdf') and not filename.startswith('.'):
                book_path = os.path.join(self.books_dir, filename)
                book_title = filename.replace('.pdf', '').replace('_', ' ')
                
                # Read PDF
                with open(book_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        # Clean up extra whitespace while preserving paragraph breaks
                        page_text = '\n\n'.join(
                            para.strip() 
                            for para in page_text.split('\n\n') 
                            if para.strip()
                        )
                        text += page_text + '\n\n'
                
                # Split into paragraphs
                paragraphs = [
                    para.strip() 
                    for para in text.split('\n\n') 
                    if para.strip() and len(para.strip().split()) > 20  # Only keep paragraphs with >20 words
                ]
                
                # Store paragraphs with book titles
                self.passages.extend([(para, book_title) for para in paragraphs])
                self.book_titles.append(book_title)
        
        # Create vector store
        if self.passages:
            passages_only = [p[0] for p in self.passages]
            embeddings = self.model.encode(passages_only)
            
            # Initialize FAISS index
            dimension = embeddings.shape[1]
            self.vector_store = faiss.IndexFlatL2(dimension)
            self.vector_store.add(embeddings.astype('float32'))
    
    def play_game(self):
        """Play the quote guessing game."""
        if not self.passages:
            print("No books loaded! Please check the books directory.")
            return
            
        print("\nWelcome to the Book Quote Guessing Game!")
        print("I'll show you a quote, and you try to guess which book it's from.")
        print("\nAvailable books:")
        for i, title in enumerate(self.book_titles, 1):
            print(f"{i}. {title}")
            
        while True:
            quote, correct_book = self.get_random_quote()
            print("\n" + "="*50)
            print("\nQuote:")
            print(f'"{quote}"')
            print("\nGuess the book (enter the number):")
            
            for i, title in enumerate(self.book_titles, 1):
                print(f"{i}. {title}")
                
            try:
                guess = int(input("\nYour guess (or 0 to quit): "))
                if guess == 0:
                    print("\nThanks for playing!")
                    break
                    
                if 1 <= guess <= len(self.book_titles):
                    guessed_book = self.book_titles[guess-1]
                    if guessed_book == correct_book:
                        print("\nCorrect! Well done!")
                    else:
                        print(f"\nSorry, that's incorrect. The quote was from '{correct_book}'")
                else:
                    print("\nInvalid choice. Please enter a valid book number.")
            except ValueError:
                print("\nPlease enter a valid number.")

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the game
game = QuoteGame()

@app.route('/api/books', methods=['GET'])
def get_books():
    """Return list of available books."""
    return jsonify({
        'books': [
            {'id': i + 1, 'title': title}
            for i, title in enumerate(game.book_titles)
        ]
    })

@app.route('/api/quote', methods=['GET'])
def get_quote():
    """Get a random quote."""
    try:
        quote, correct_book = game.get_random_quote()
        # Store the correct book in the session
        return jsonify({
            'quote': quote,
            'correct_book': correct_book,  # We'll send this for demo purposes, but in a real game you might want to keep it server-side
            'books': [
                {'id': i + 1, 'title': title}
                for i, title in enumerate(game.book_titles)
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/check', methods=['POST'])
def check_answer():
    """Check if the guessed book is correct."""
    data = request.get_json()
    if not data or 'guess' not in data or 'correct_book' not in data:
        return jsonify({'error': 'Invalid request'}), 400
    
    guess = data['guess']
    correct_book = data['correct_book']
    
    # Debug logging
    print(f"Received guess: '{guess}' (type: {type(guess)})")
    print(f"Correct book: '{correct_book}' (type: {type(correct_book)})")
    
    # Normalize strings for comparison
    guess = str(guess).strip().lower()
    correct_book = str(correct_book).strip().lower()
    
    is_correct = guess == correct_book
    print(f"Comparison result: {is_correct}")
    
    if is_correct:
        return jsonify({
            'status': 'success',
            'message': 'Correct! Well done!',
            'isCorrect': True
        })
    else:
        return jsonify({
            'status': 'incorrect',
            'message': f"Sorry, that's not correct. The correct answer was '{correct_book}'",
            'isCorrect': False
        })

if __name__ == "__main__":
    app.run(debug=True, port=5000)

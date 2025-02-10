import requests
import time

# Base URL
base_url = "http://localhost:8000"

def test_flow():
    # Get a quote
    print("\n" + "="*50)
    print("Getting a quote...")
    quote_response = requests.get(f"{base_url}/quote")
    quote_data = quote_response.json()
    print("\nQUOTE:")
    print("-"*50)
    print(quote_data.get('quote'))
    print("-"*50)
    print(f"Page: {quote_data.get('page')}")
    
    # Print image path if available
    if 'image_path' in quote_data:
        print(f"\nPage image saved as: {quote_data['image_path']}")
    
    # Wait for user input with clear prompt
    print("\n" + "="*50)
    guess = input("\nWhat book is this quote from? Enter your guess: ")
    
    # Evaluate the guess
    data = {
        "guess": guess,
        "correct_book": quote_data['correct_book']['title']
    }
    eval_response = requests.post(f"{base_url}/evaluate", json=data)
    eval_data = eval_response.json()
    
    print("\nRESULT:")
    print("-"*50)
    print("Feedback:", eval_data.get('feedback'))
    print("Correct?", eval_data.get('is_correct'))
    print("="*50)

if __name__ == "__main__":
    print("\nWelcome to the Quote Game!")
    while True:
        test_flow()
        again = input("\nWould you like to try another quote? (yes/no): ")
        if again.lower() != 'yes':
            print("\nThanks for playing!")
            break 
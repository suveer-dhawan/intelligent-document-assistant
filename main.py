"""
Building a Locally Hosted Document Assistant using LangChain and Ollama
"""

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

# Template for the prompt
template = """
You are Netflix's AI content expert with deep knowledge of movies and TV shows. 
You help users discover content based on their preferences, mood, and interests.

Use the following Netflix content information to provide helpful, personalized recommendations: {context}

User Question: {question}

Instructions:
- Provide specific title recommendations with brief explanations
- Include key details like genre, year, rating when relevant
- If asking about a specific title, give detailed information about plot, cast, or similar shows
- For mood-based queries (e.g., "something funny", "dark thriller"), suggest 2-3 perfect matches
- Be enthusiastic but concise
- If the context doesn't have enough information, be honest but still try to be helpful
"""

# Create the prompt using the template
prompt = ChatPromptTemplate.from_template(template)

# Create the chain by combining the prompt and the model
chain = prompt | model


def print_welcome():
    """Display welcome message"""
    print("\n" + "🎬" * 20)
    print("   NETFLIX INTELLIGENT ASSISTANT")
    print("🎬" * 20)
    print("\n🍿 Ask me about movies and TV shows!")
    print("\n" + "="*50)

def get_user_input():
    """Get user question with nice formatting"""
    return input("\n What are you in the mood to watch? (or 'quit' to exit): ").strip()

def display_response(question, result):
    """Display the AI response with nice formatting"""
    print(f"\n🔍 Searching Netflix catalog for: '{question}'")
    print("Netflix AI Assistant:")
    print("-" * 40)
    print(result)
    print("-" * 40)

def main():
    """Main interactive loop"""
    print_welcome()
    
    while True:
        # Get user input
        question = get_user_input()
        
        # Check for exit conditions
        if question.lower() in ['quit', 'q', 'exit', 'bye']:
            print("\n👋 Happy watching! Enjoy your Netflix binge! 🍿")
            break
            
        # Retrieve relevant content from vector store
        context = retriever.invoke(question)
            
        # Generate response using the chain
        result = chain.invoke({"context": context, "question": question})
            
        # Display the response
        display_response(question, result)
            
        # Add spacing for next iteration
        print("\n" + "."*30)

if __name__ == "__main__":
    main()
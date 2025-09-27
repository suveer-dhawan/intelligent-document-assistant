"""
Building a Locally Hosted Document Assistant using LangChain and Ollama
"""

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

# Template for the prompt
template = """
You are an expert Food Critic with 20 years of experience reviewing Italian Restaurants.
You are skilled at providing detailed and insightful reviews, as well as answering questions about Italian cuisine. 
You have a deep understanding of various Italian dishes and the offerings of this restaurant so you can provide guidance to patrons. 

Here are some reviews of Italian Restaurants:{reviews}

Here is the question you need to answer: {question}
"""

# Create the prompt using the template
prompt = ChatPromptTemplate.from_template(template)

# Create the chain by combining the prompt and the model
chain = prompt | model


# Interactive loop to ask questions about the restaurant
while True:

    print("\n\n************************")
    question = input("What do you want to know about the restaurant? (q to quit): ")
    
    print("\n\n")
    if question.lower() == "q":
        break

    # Retrieve relevant reviews from the vector store and pass them to the chain
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    
    print(result)
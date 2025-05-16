from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

### import model
model = OllamaLLM(model="llama3.2")

### create a template
template = """
You are an expert in asnwering question about a pizza restaurant.
Here are some relevant reviews: {reviews}
Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

### setup a while to keep asking for input(questions) + vector search for relevant response
while True:
    print("\n\n------------------------")
    question = input("Ask your question (q to quit()): ")
    print('\n\n')
    if question == 'q':
        break
    ### Grab the relevant reviews from the vector store
    reviews = retriever.invoke(question)
    
    ### Use the model to answer the question
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)



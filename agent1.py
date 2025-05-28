import os
import openai
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.tools import tool

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define a tool for handling user input
@tool
def user_response(prompt: str) -> str:
    """
    Generates a response using OpenAI GPT based on the provided prompt.
    Catches errors to prevent tool usage failures.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error retrieving response: {str(e)}"

# Create an AI agent
chatbot_agent = Agent(
    role="AI Chat Assistant",
    goal="Engage in conversations and provide helpful responses.",
    backstory="A chatbot designed to answer user questions intelligently.",
    tools=[user_response]  # Assign the tool
)

# Define a task for the chatbot
def create_chat_task(user_input: str):
    return Task(
        description=f"Respond to the following user query: '{user_input}'",
        agent=chatbot_agent,
        expected_output="A relevant and conversational answer to the user's question."
    )

# Function to interact with the chatbot
def chat():
    print("Chatbot: Hello! How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        # Dynamically create a new task for each user input
        chat_task = create_chat_task(user_input)

        # Create a new Crew instance for each interaction
        chat_crew = Crew(agents=[chatbot_agent], tasks=[chat_task])

        # Kickoff the Crew and retrieve the response
        try:
            response = chat_crew.kickoff()
            if isinstance(response, list):  # CrewAI might return a list
                response = response[0]  # Take the first response
        except Exception as e:
            response = f"Error: {str(e)}"

        print(f"Chatbot: {response}")

# Run the chatbot
if __name__ == "__main__":
    chat()

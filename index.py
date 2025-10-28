import os
import random
import requests
import gradio as gr
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

def load_llm():
    llm = ChatOllama(model="llama3.2", temperature=0)
    print("Ollama LLM loaded successfully!")
    return llm

# load .env
load_dotenv()

# Load LLM
llm = load_llm()

# Define Tools
@tool
def get_weather(city: str) -> str:
    """Fetches current weather for a city using Tomorrow.io API."""
    API_KEY = os.getenv('TOMORROW_API_KEY')
    
    try:
        geo_url = f"https://api.tomorrow.io/v4/locations/search"
        geo_params = {
            "text": city,
            "limit": 1,
            "apikey": API_KEY
        }
        geo_response = requests.get(geo_url, para=geo_params).json()
        if not geo_response.get('data') or not geo_response['data']:
            return f"Could not find location for {city}. Try a different spelling."
        location_data = geo_response['data'][0]
        lat, lon = location_data['latitude'], location_data['longitude']
        
        weather_url = f"https://api.tomorrow.io/v4/weather/realtime"
        weather_params = {
            "location": f"{lat},{lon}",
            "units": "metric",
            "apikey": API_KEY
        }
        
        weather_response = requests.get(weather_url, params=weather_params).json()
        
        if 'data' not in weather_response or 'values' not in weather_response['data']:
            return ValueError("Invalid weather data received.")
        
        current = weather_response["data"]["values"]
        temp = current["temperature"]
        windspeed = current["windspeed"]
        weather_code = current.get("weatherCode", "unknown")
        
        conditions = {
            "clear": "Clear skies",
            "partly-cloudy-day": "Partly cloudy",
            "rain": "Rain",
            "fog": "Foggy",
            "light-rain": "Light rain",
        }
        condition = conditions.get(weather_code, f"COnditions: {weather_code}")
        
        return f"Weather in {city} ({lat:.2f}°{'S' if lat < 0 else 'N'}, {lon:.2f}°{'W' if lon < 0 else 'E'}): {temp}°C, {condition}, wind {windspeed} km/h."
    except Exception as e:
        temps = [10, 15, 20, 25]
        conditions = ["sunny", "rainy", "cloudy", "windy"]
        temp = random.choice(temps)
        cond = random.choice(conditions)
        return f"Error fetching weather: Mock for {city}: {temp}°C, {cond}."

@tool
def suggest_activities(city: str, weather_desc: str) -> str:
    """Suggests 3 activities based on city and weather."""
    prompt = f"Suggest 3 fun, weather-appropriate activities for {city} given: {weather_desc}. Number them 1-3, keep concise."
    return llm.invoke(prompt).content

@tool
def book_flight(origin: str, destination: str, date: str) -> str:
    """Mocks booking a flight."""
    return f"Flight booked: {origin} → {destination} on {date}. Confirmation: FLIGHT-{random.randint(10000, 99999)}."

tools = [get_weather, suggest_activities, book_flight]

global_agent_config = {"configurable": {"thread_id": "web_session_1"}}

# Build the Agent 
def build_agent():
    checkpointer = MemorySaver()
    system_prompt = """You are an autonomous trip planning agent. For any query:
    - Step 1: Decompose into sub-tasks (e.g., weather → activities → booking).
    - Step 2: Call tools only for needed info; reason aloud.
    - Step 3: Synthesize a clear plan/response.
    Be helpful, concise, and proactive. If unclear, ask for details."""
    app = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer
    )
    return app, checkpointer

app, checkpointer = build_agent()  # Unchanged

# New: Gradio chat function
def chat_with_agent(message, history, debug_mode):
    """
    Handles one chat turn: Invoke agent, format response (Gradio auto-appends to history).
    - message: Current user input (str)
    - history: List of past [[user_msg, bot_msg], ...] (for reference, but don't modify)
    - debug_mode: Bool from checkbox
    """
    # Prepare input for agent (unchanged)
    input_message = {"messages": [HumanMessage(content=message)]}
    
    # Invoke agent with persistent config (unchanged)
    response = app.invoke(input_message, global_agent_config)
    agent_reply = response['messages'][-1].content
    
    # Mock reasoning (unchanged, but now baked into reply str)
    reasoning = ""
    if debug_mode:
        reasoning = "Thought: Decomposed query → Called get_weather → Synthesized plan."
        agent_reply += f"\n\n<details><summary>Agent Thoughts</summary>{reasoning}</details>"
    
    # FIXED: Return ONLY the reply str—Gradio handles history append as [[message, agent_reply]]
    return agent_reply  # Just str! No (history, "")

def create_gradio_interface():
    # Debug checkbox
    debug_checkbox = gr.Checkbox(label="Show Agent Reasoning (Verbose)", value=False)
    
    # Chat interface: Uses our chat function
    chatbot = gr.ChatInterface(
        fn=chat_with_agent,
        title="Trip Planning Agent",
        description="Ask about weather, activities, or book flights! E.g., 'Weather in Paris?'",
        examples=[
            ["What's the weather like in Tokyo?", False],
            ["Suggest activities in New York if it's sunny.", False],
            ["Book a flight from LA to Miami on Dec 20.", False],
        ],
        cache_examples=False,
        additional_inputs=[debug_checkbox],
        theme=gr.themes.Soft(),
    )
    return chatbot

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
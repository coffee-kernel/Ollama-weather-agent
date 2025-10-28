import os
import random
import requests
import streamlit as st
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

# streamlit page config
st.set_page_config(page_title="Ollama Trip Planner", page_icon="✈️", layout="wide")
st.title("✈️ Ollama Trip Planner Agent with LangGraph v1.0")
st.markdown("Chat with an autonomous agent that plans trips using Ollama LLM and LangGraph v1.0!")

@st.cache_resource
def load_llm():
    llm = ChatOllama(model="llama3.2", temperature=0)
    st.success("Ollama LLM loaded successfully!")
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

# Build the Agent (Cached)
@st.cache_resource
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

app, checkpointer = build_agent()

# Step 4: Streamlit Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []
if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": "web_session_1"}}

# Sidebar for debug
debug = st.sidebar.checkbox("Show Agent Reasoning (Verbose)")
st.sidebar.markdown("**Tips:** Refresh to reset chat. Ensure Ollama is running.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if debug and message.get("reasoning"):
            with st.expander("Agent Thoughts"):
                st.markdown(message["reasoning"])

# Chat input
if prompt := st.chat_input("Ask about a trip, e.g., 'Weather in Paris? Plan a sunny day...'"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Planning your trip..."):
            input_message = {"messages": [HumanMessage(content=prompt)]}
            response = app.invoke(input_message, st.session_state.config)
            agent_reply = response['messages'][-1].content
            
            reasoning = ""
            if debug:
                reasoning = "Thought: Decomposed query → Called get_weather → Synthesized plan."
            
            st.markdown(agent_reply)
            if debug and reasoning:
                with st.expander("Agent Thoughts"):
                    st.markdown(reasoning)
        
        # Store in session
        st.session_state.messages.append({
            "role": "assistant", 
            "content": agent_reply,
            "reasoning": reasoning if debug else None
        })
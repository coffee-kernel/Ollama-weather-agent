import os
import random
import requests
import streamlit as st
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama  # Updated to ChatOllama for better agent compat
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent  # <-- FIXED: New import for v1.0

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

llm = load_llm()

# Step 2: Define Tools (Unchanged)
@tool
def get_weather(city: str) -> str:
    """Fetches current weather for a city using Open-Meteo (free, no key)."""
    try:
        geo_url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
        geo_headers = {"User-Agent": "WeatherAgent/1.0"}
        geo_response = requests.get(geo_url, headers=geo_headers).json()
        if not geo_response:
            return f"Could not find location for {city}. Try a different spelling."
        lat, lon = float(geo_response[0]["lat"]), float(geo_response[0]["lon"])
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&temperature_unit=celsius&wind_speed_unit=kmh&timezone=auto"
        weather_response = requests.get(weather_url).json()
        current = weather_response["current_weather"]
        temp = current["temperature"]
        windspeed = current["windspeed"]
        weather_code = current["weather_code"]
        
        conditions = {0: "Clear skies", 1: "Mainly clear", 3: "Rain", 45: "Fog", 61: "Light rain"}
        condition = conditions.get(weather_code, "Partly cloudy")
        
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
    return f"Flight booked: {origin} → {destination} on {date}. Confirmation: FLIGHT-{random.randint(10000, 99999)}. Cost: ~$400 (mock)."

tools = [get_weather, suggest_activities, book_flight]

# Step 3: Build the Agent (Cached)
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
    st.session_state.messages = []  # Local chat history for display
if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": "web_session_1"}}  # LangGraph thread for memory

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
            
            # Optional: Capture reasoning (if verbose/debug enabled)
            reasoning = ""
            if debug:
                # Simulate verbose (in real, log from graph; here, mock for demo)
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
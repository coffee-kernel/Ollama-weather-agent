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

app, checkpointer = build_agent()

# New: Gradio chat function
def chat_with_agent(message, history, debug_mode, edit_mode=False):
    """
    Handles chat: Append for normal, replace last [user, bot] for edit.
    Returns updated history (Gradio displays it).
    """
    # Prepare input for agent 
    input_message = {"messages": [HumanMessage(content=message)]}
    
    # Invoke agent with persistent config 
    response = app.invoke(input_message, global_agent_config)
    agent_reply = str(response['messages'][-1].content)
    
    # Add reasoning if debug 
    reasoning = ""
    if debug_mode:
        reasoning = "Thought: Decomposed query → Called get_weather → Synthesized plan."
        agent_reply += f"\n\n<details><summary>Agent Thoughts</summary>{reasoning}</details>"
    
    if edit_mode and history and len(history) > 0:
        history[-1] = [message, agent_reply]
    else:
        history.append([message, agent_reply])
    
    return history

def create_gradio_interface():
    with gr.Blocks(title="Trip Planning Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Trip Planning Agent")
        gr.Markdown("Ask about weather, activities, or book flights! E.g., 'Weather in Paris?'")
        
        # Debug checkbox
        debug_checkbox = gr.Checkbox(label="Show Agent Reasoning (Verbose)", value=False)
        
        # Hidden state for edit mode
        editing_state = gr.State(False)
        
        # Chatbot for history display
        chatbot = gr.Chatbot(
            height=400,
            show_label=False,
            avatar_images=(
                "./Assets/user_avatar.png",
                "./Assets/bot.png"
            )
        )
        
        with gr.Row():
            msg_input = gr.Textbox(
                label="Your Message",
                placeholder="Type here...",
                scale=4,
                show_label=False
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
            edit_btn = gr.Button("Edit Last", visible=False, scale=1)
            clear_btn = gr.Button("Clear Chat", variant="secondary", scale=1)
        
        gr.Examples(
            examples=[
                ["Weather in Paris?"],
                ["Plan a sunny day in Tokyo"],
                ["Book flight from NYC to LA on Oct 30"]
            ],
            inputs=[msg_input],
            label="Quick Starts"
        )
        
        def process_message(msg, history, debug, editing):
            if not msg.strip():
                return history, "", gr.update(visible=bool(history)), False
            # Call agent with edit_mode
            updated_history = chat_with_agent(msg, history, debug, edit_mode=editing)
            new_edit_visible = gr.update(visible=bool(updated_history))
            new_edit_btn = gr.update(value="Edit Last") if not editing else gr.update(value="Edit Last")
            return updated_history, "", new_edit_visible, new_edit_btn, False  # Reset editing
        
        send_btn.click(
            process_message,
            inputs=[msg_input, chatbot, debug_checkbox, editing_state],
            outputs=[chatbot, msg_input, edit_btn, edit_btn, editing_state]  # Last two for btn update + state reset
        )
        msg_input.submit(
            process_message,
            inputs=[msg_input, chatbot, debug_checkbox, editing_state],
            outputs=[chatbot, msg_input, edit_btn, edit_btn, editing_state]
        )
        
        # Event: Edit Last button
        def start_edit(history):
            if not history or len(history) == 0:
                return history, "", gr.update(visible=False), False
            last_user_msg = history[-1][0]
            return history, last_user_msg, gr.update(value="Update", visible=True), True
        
        edit_btn.click(
            start_edit,
            inputs=[chatbot],
            outputs=[chatbot, msg_input, edit_btn, editing_state]
        )
        
        # Event: Clear chat
        def clear_all():
            return [], "", gr.update(visible=False), False
        
        clear_btn.click(
            clear_all,
            outputs=[chatbot, msg_input, edit_btn, editing_state]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
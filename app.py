import openai
import pandas as pd
import streamlit as st
import os
 
# Set your OpenAI API key
client = openai.OpenAI(api_key= os.getenv("OPENAI_API_KEY"))
 
def load_data():
    # Replace with your dataset path
    data = pd.read_csv("cleaned_data.csv")
    # Precompute insights
    insights = {
        # Basic statistics
        "average_traffic_volume": data['Traffic Volume'].mean(),
        "average_passenger_count": data['Passenger Count'].mean(),
        "average_noise_level": data['Noise Level'].mean(),
        "average_speed": data['Average Speed'].mean(),
        "average_pm25": data['PM2.5 Level'].mean(),
        "average_aqi": data['AQI'].mean(),
 
        # Grouped insights
        "traffic_by_location": data.groupby('Location')['Traffic Volume'].mean().to_dict(),
        "passenger_by_location": data.groupby('Location')['Passenger Count'].mean().to_dict(),
        "noise_by_location": data.groupby('Location')['Noise Level'].mean().to_dict(),
        "pm25_by_location": data.groupby('Location')['PM2.5 Level'].mean().to_dict(),
        "aqi_by_location": data.groupby('Location')['AQI'].mean().to_dict(),
        "speed_by_location": data.groupby('Location')['Average Speed'].mean().to_dict(),
 
        # Time-based insights
        "aqi_by_time": data.groupby('Time of the Day')['AQI'].mean().to_dict(),
        "noise_by_time": data.groupby('Time of the Day')['Noise Level'].mean().to_dict(),
        "speed_by_time": data.groupby('Time of the Day')['Average Speed'].mean().to_dict(),
 
        # Correlation matrix (only numeric columns)
        "correlation_matrix": data.select_dtypes(include=['float64', 'int64']).corr().to_dict(),
 
        # Data types
        "data_types": data.dtypes.to_dict()
    }
    return data, insights
 
# Define the system prompt
system_prompt = """
You are an AI assistant for an urban mobility hackathon. Your task is to provide recommendations, insights, and guidance based on the dataset and tasks provided. Do not provide specific numerical values or generate code. If a question is unrelated to the dataset or tasks, respond with: "I can only answer questions related to the urban mobility dataset and hackathon tasks."
 
Dataset Description:
- The dataset contains information about urban mobility, including traffic volume, passenger count, noise levels, air quality, and more.
- Features: ['Location', 'Traffic Volume', 'Passenger Count', 'Noise Level', 'Average Speed', 'PM2.5 Level', 'AQI', 'Day of the Week', 'Time of the Day']
- Categorical Columns: ['Location', 'Day of the Week', 'Time of the Day']
 
Hackathon Tasks:
1. Data Analysis:
   - Import the dataset.
   - Summarize the dataset.
   - Create a correlation matrix heatmap.
   - Create a boxplot of the dataset.
   - Remove outliers using the capping method.
   - Fill missing data using suitable methods.
   - Change column datatypes.
 
2. Advanced Analysis:
   - Calculate the average traffic volume for locations with a passenger count above 30,000.
   - Identify locations with the highest and lowest average speed.
   - Perform environmental correlation analysis.
   - Group locations by 'Time of the Day' and calculate average AQI.
   - Calculate traffic density (traffic volume / average speed) for all locations.
   - Determine the most/least active locations by Traffic Volume and Passenger Count.
 
3. AI Model:
   - Preprocess data for machine learning (encode categorical data, scale feature columns).
   - Prepare the dataset for model training (feature-target separation).
   - Train machine learning models (regression or classification).
 
**Important Rules**:
1. Provide recommendations and insights, not specific numerical values.
2. Do not generate or suggest code.
3. Do not answer questions unrelated to the dataset or hackathon tasks.
4. Always base your answers on the dataset and tasks provided.
5. Respond to greetings and small talk politely.
"""
 
# Function to ask the chatbot
def ask_chatbot(question, insights):
    # List of greetings and small talk phrases
    greetings = ["hello", "hi", "hey", "how are you", "good morning", "good afternoon", "good evening"]
 
    # Check if the question is a greeting
    is_greeting = any(greeting in question.lower() for greeting in greetings) and not any(word in question.lower() for word in ["aqi", "traffic", "noise", "speed", "pm2.5", "passenger", "correlation", "dataset"])
 
    # If it's a greeting, respond politely
    if is_greeting:
        return "Hello! I'm here to help you with the urban mobility dataset and hackathon tasks. How can I assist you today?"
 
    # Combine system prompt and user question
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
 
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,  # Increase token limit for detailed responses
        temperature=0.7
    )
 
    # Extract the chatbot's response
    chatbot_response = response.choices[0].message.content
 
    # Add precomputed insights if relevant
    if "traffic volume at" in question.lower():
        location = question.split("at")[-1].strip()
        traffic_volume = insights["traffic_by_location"].get(location, "Location not found.")
        chatbot_response += f"\nInsight: Traffic volumes at {location} are generally higher in urban areas compared to suburban areas."
    elif "highest aqi" in question.lower():
        highest_aqi_location = max(insights["aqi_by_location"], key=insights["aqi_by_location"].get)
        chatbot_response += f"\nInsight: The location with the highest AQI is typically an industrial or high-traffic area, indicating poor air quality."
    elif "high correlation" in question.lower():
        correlation_matrix = insights["correlation_matrix"]
        high_corr_pairs = [(col1, col2, corr) for col1 in correlation_matrix for col2, corr in correlation_matrix[col1].items() if col1 != col2 and abs(corr) > 0.7]
        chatbot_response += f"\nInsight: Strong correlations exist between certain variables, such as traffic volume and noise levels. These relationships can help identify patterns in urban mobility."
    elif "data types" in question.lower():
        data_types = insights["data_types"]
        chatbot_response += f"\nInsight: The dataset contains a mix of numerical and categorical data. Numerical columns include traffic volume and AQI, while categorical columns include location and time of day."
    elif "highest pm2.5 level" in question.lower():
        highest_pm25_location = max(insights["pm25_by_location"], key=insights["pm25_by_location"].get)
        chatbot_response += f"\nInsight: The location with the highest PM2.5 levels is likely an industrial or densely populated area, indicating higher pollution levels."
    elif "highest noise level during the night" in question.lower():
        noise_at_night = insights["noise_by_time"].get("Night", "No data available for night.")
        highest_noise_location = max(insights["noise_by_location"], key=insights["noise_by_location"].get)
        chatbot_response += f"\nInsight: Noise levels during the night are typically higher in areas with active nightlife or heavy traffic."
    elif "aqi above 400" in question.lower():
        high_aqi_locations = [loc for loc, aqi in insights["aqi_by_location"].items() if aqi > 400]
        chatbot_response += f"\nInsight: Locations with AQI above 400 are rare but indicate extremely poor air quality, often due to industrial emissions or wildfires."
    elif "lowest pm2.5 level and aqi" in question.lower():
        lowest_pm25_location = min(insights["pm25_by_location"], key=insights["pm25_by_location"].get)
        lowest_aqi_location = min(insights["aqi_by_location"], key=insights["aqi_by_location"].get)
        chatbot_response += f"\nInsight: Locations with the lowest PM2.5 levels and AQI are typically residential or rural areas with minimal pollution sources."
 
    return chatbot_response
 
# Streamlit app
def main():
    st.title("Urban Mobility Chatbot")
    st.write("Ask questions about the urban mobility dataset and hackathon tasks.")
 
    # Custom CSS to style the sidebar
    st.markdown(
        """
        <style>
        /* Sidebar background and text color */
        .css-1d391kg {
            background-color: #e8f8f5;  /* Light gray background */
            padding: 20px;
            border-radius: 10px;
        }
        /* Sidebar header text color */
        .css-1d391kg h1 {
            color: #2e86c1;  /* Blue header text */
        }
        /* Sidebar body text color */
        .css-1d391kg p, .css-1d391kg ul {
            color: #34495e;  /* Dark gray text */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
 
    # Sidebar for instructions
    with st.sidebar:
        st.title("How to Use the Chatbot")
        st.write("""
        **Welcome to the Urban Mobility Chatbot!**  
        This chatbot is designed to help you analyze and understand the urban mobility dataset. Here's how you can use it:
 
        - **Ask Questions**: Type your question in the input box below and click "Submit".
        - **Dataset Insights**: The chatbot can provide insights about traffic volume, air quality, noise levels, and more.
        - **Hackathon Tasks**: Use the chatbot to get guidance on tasks like data analysis, advanced analysis, and AI modeling.
 
        **Example Questions**:
        - What is the average traffic volume?
        - Which location has the highest AQI?
        - What are the high correlations in the dataset?
        """)
 
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
 
    # Load data and precompute insights
    data, insights = load_data()
 
    # Input box for user question
    user_question = st.text_input("Enter your question:")
 
    # Submit button
    if st.button("Submit"):
        if user_question:
            # Add the user's question to the chat history first
            st.session_state.chat_history.append({"role": "user", "content": user_question})
 
            # Get chatbot response
            response = ask_chatbot(user_question, insights)
 
            # Add the chatbot's response to the chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
 
    # Display chat history (newer messages first, user over chatbot)
    st.write("Chat History:")
    for i in range(len(st.session_state.chat_history) - 1, -1, -2):  # Iterate in reverse steps of 2
        if i >= 0:
            user_message = st.session_state.chat_history[i - 1]  # User message
            chatbot_message = st.session_state.chat_history[i]  # Chatbot message
 
            # Display user message with unique identifier
            st.write(f"**YOU**: {user_message['content']}")
            # Display chatbot message with unique identifier
            st.write(f"**Chatbot**: {chatbot_message['content']}")
# Run the app
if __name__ == "__main__":
    main()
 

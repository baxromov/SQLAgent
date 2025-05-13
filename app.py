import streamlit as st
from urllib.parse import quote
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
import os

# Set up Streamlit UI
st.title("PostgreSQL Database Query Assistant")

# Obtain API Key (for LLM)
api_key = st.text_input("Enter your Cohere API Key", type="password", key="api_key")
if api_key:
    os.environ["CO_API_KEY"] = api_key

# Initialize LLM
if api_key:
    llm = init_chat_model("cohere:command-r-plus")
else:
    st.warning("Please provide a Cohere API Key to proceed.")
    st.stop()

# Database credentials inputs
st.subheader("Enter Your PostgreSQL Database Credentials")
username = st.text_input("Username", key="username")
password = st.text_input("Password (Special characters will be handled)", type="password", key="password")
host = st.text_input("Hostname (e.g., localhost)", key="host")
port = st.text_input("Port (e.g., 5432)", value="5432", key="port")
database = st.text_input("Database Name (e.g., mydatabase)", key="database")

# Proceed if all fields are filled
if username and password and host and port and database:
    try:
        # URL-encode the password to handle special characters
        encoded_password = quote(password)

        # Construct the `db_uri`
        db_uri = f"postgresql://{username}:{encoded_password}@{host}:{port}/{database}"

        # Display the constructed URI
        # Note: Omitting the password display for security
        st.write(f"Constructed Database URI: `{db_uri}` (password is encoded for connection)")

        # Connect to the database
        db = SQLDatabase.from_uri(db_uri)
        st.success("Connected to the database successfully!")

        # Setup the SQL toolkit
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()

        # Optional: Display available tools
        if st.checkbox("Show Available Tools"):
            for tool in tools:
                st.text(f"{tool.name}: {tool.description}")

        # Define the system prompt for SQL interaction
        system_prompt = """
        You are an assistant designed to interact with a SQL database.
        Given an input question, create a syntactically correct SQL query to run,
        then look at the results of the query and return the answer. Unless the user
        specifies a specific number of examples they wish to obtain, always limit your
        query to at most {top_k} results.
        You can order the results by a relevant column to return the most interesting
        examples in the database. Never query for all the columns from a specific table,
        only ask for the relevant columns given the question.
        You MUST double check your query before executing it. If you get an error while
        executing a query, rewrite the query and try again.
        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
        database.
        When writing queries, ALWAYS wrap table names with double quotes (""). This is
        especially important for table names that include reserved keywords or special
        characters.
        To start, ALWAYS look at the tables in the database to see what you can query.
        DO NOT skip this step. Then query the schema of the most relevant tables.
        """.format(dialect=db.dialect, top_k=5)

        # Create the query agent
        agent = create_react_agent(
            llm,
            tools,
            prompt=system_prompt,
        )

        # User input for querying
        st.markdown("---")
        st.subheader("Ask Your Database a Question")
        user_question = st.text_input("Ask a question (e.g., What are the top 5 customers?)")

        if user_question:
            with st.spinner("Processing your question..."):
                try:
                    # Stream the agent's response
                    for step in agent.stream({"messages": [{"role": "user", "content": user_question}]}):
                        # Ensure step object contains expected data
                        if "agent" in step and "messages" in step["agent"]:
                            messages = step["agent"]["messages"]
                            if isinstance(messages, list) and len(messages) > 0:
                                # Safely access the content of the last message
                                message = messages[-1].get("content", "(No content in response)")
                                st.write(message)
                            else:
                                st.error("The response messages are in an unexpected format.")
                        else:
                            st.error("Unexpected structure in the streamed response.")

                except Exception as e:
                    st.error(f"An error occurred while processing your question: {e}")

    except Exception as connection_error:
        st.error(f"Failed to connect to the database: {connection_error}")

else:
    st.warning("Please provide all required fields to construct the database URI.")
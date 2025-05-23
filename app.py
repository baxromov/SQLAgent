import os
from urllib.parse import quote

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.prebuilt import create_react_agent

st.title("PostgreSQL Database Query Assistant")

api_key = st.text_input("Enter your Cohere API Key", type="password", key="api_key")
if api_key:
    os.environ["CO_API_KEY"] = api_key

if api_key:
    llm = init_chat_model("cohere:command-r-plus")
else:
    st.warning("Please provide a Cohere API Key to proceed.")
    st.stop()

st.subheader("Enter Your PostgreSQL Database Credentials")
username = st.text_input("Username", key="username")
password = st.text_input("Password (Special characters will be handled)", type="password", key="password")
host = st.text_input("Hostname (e.g., localhost)", key="host")
port = st.text_input("Port (e.g., 5432)", value="5432", key="port")
database = st.text_input("Database Name (e.g., mydatabase)", key="database")

if username and password and host and port and database:
    try:
        encoded_password = quote(password)

        db_uri = f"postgresql://{username}:{encoded_password}@{host}:{port}/{database}"
        st.write(f"Constructed Database URI: `{db_uri}` (password is encoded for connection)")

        db = SQLDatabase.from_uri(db_uri)
        st.success("Connected to the database successfully!")

        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()

        if st.checkbox("Show Available Tools"):
            for tool in tools:
                st.text(f"{tool.name}: {tool.description}")

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

        agent = create_react_agent(
            llm,
            tools,
            prompt=system_prompt,
        )

        st.markdown("---")
        st.subheader("Ask Your Database a Question")
        user_question = st.text_input("Ask a question (e.g., What are the top 5 customers?)")

        if user_question:
            with st.spinner("Processing your question..."):
                try:
                    for step in agent.stream(
                            {"messages": [{"role": "user", "content": user_question}]},
                            stream_mode="values",
                    ):
                        st.write(step["messages"][-1].pretty_print())


                except Exception as e:
                    st.error(f"An error occurred while processing your question: {e}")

    except Exception as connection_error:
        st.error(f"Failed to connect to the database: {connection_error}")

else:
    st.warning("Please provide all required fields to construct the database URI.")

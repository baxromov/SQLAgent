from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
import os

# Set API key for the LLM
if not os.environ.get("CO_API_KEY"):
    os.environ["CO_API_KEY"] = "tfCACiGTo94egdP6vDzPH2IP4w4HR6e4H7jRVahC"

from langchain.chat_models import init_chat_model

# Initialize the LLM
llm = init_chat_model("cohere:command-r-plus")

# Connect to PostgreSQL database
db = SQLDatabase.from_uri("postgresql://postgres:postgres@localhost:5432/supplychain-local")
# print(f"Dialect: {db.dialect}")
# print(f"Available tables: {db.get_usable_table_names()}")
# print(f'Sample output: {db.run('SELECT * FROM "order" LIMIT 5;')}')

# Create a toolkit for interacting with the SQL database
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
#
# # Print out tool descriptions
for tool in tools:
    print(f"{tool.name}: {tool.description}\n")
#
from langgraph.prebuilt import create_react_agent
#
# # System prompt with PostgreSQL dialect
system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
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
When writing queries, ALWAYS wrap table names with double quotes (""). For
example, instead of writing SELECT * FROM table_name, write SELECT * FROM
"table_name". This is especially important for table names that include reserved
keywords or special characters.
To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.
Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

agent = create_react_agent(
    llm,
    tools,
    prompt=system_prompt,
)

question = "Which vendor has fulfilled the most orders?"
for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
):
    step["messages"][-1].pretty_print()
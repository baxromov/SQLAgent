from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
import os


if not os.environ.get("CO_API_KEY"):
    os.environ["CO_API_KEY"] = ""

from langchain.chat_models import init_chat_model


llm = init_chat_model("cohere:command-r-plus")

db = SQLDatabase.from_uri("postgresql://postgres:postgres@localhost:5432/supplychain-local")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
for tool in tools:
    print(f"{tool.name}: {tool.description}\n")
from langgraph.prebuilt import create_react_agent
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

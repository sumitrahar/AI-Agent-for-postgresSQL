from langchain.chat_models import AzureChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_openai import AzureChatOpenAI

from sqlalchemy import create_engine
import pandas as pd
import psycopg2

# PostgreSQL Connection Parameters
conn_params = {
    "host": "127.0.0.1",
    "database": "nba_db",   # ✅ Change to your actual database
    "user": "postgres",     # ✅ Replace with your PostgreSQL username
    "password": "12345",    # ✅ Your PostgreSQL password
    "port": "5432"
}

# ✅ Step 1: Test psycopg2 Connection (Raw Connection)
try:
    conn = psycopg2.connect(**conn_params)
    print("✅ Successfully connected to PostgreSQL using psycopg2!")
    conn.close()
except Exception as e:
    print("❌ Error connecting to PostgreSQL:", e)

# ✅ Step 2: Define SQLAlchemy Engine (Needed for pandas)
engine = create_engine(f"postgresql://{conn_params['user']}:{conn_params['password']}@{conn_params['host']}:{conn_params['port']}/{conn_params['database']}")

# ✅ Step 3: Fetch All Tables in the Database
query = """
SELECT table_schema, table_name 
FROM information_schema.tables 
WHERE table_schema NOT IN ('pg_catalog', 'information_schema') 
ORDER BY table_schema, table_name;
"""

df_tables = pd.read_sql(query, engine)
print(df_tables)
POSTGRES_AGENT_PREFIX = """
You are an agent designed to interact with a PostgreSQL database.

## Instructions:
- Given an input question, create a syntactically correct **PostgreSQL** query to run, then look at the results of the query and return the answer.
- Unless the user specifies a specific number of examples they wish to obtain, **ALWAYS** limit your query to at most {top_k} results using `LIMIT {top_k}`.
- You can order the results by a relevant column to return the most meaningful data.
- Never query all columns from a specific table (`SELECT *`), only retrieve the **relevant columns** based on the question.
- You have access to tools for interacting with the database.
- You **MUST double-check your query before executing it**. If you encounter an error, **rewrite the query and try again**.
- **DO NOT** modify the database (**NO INSERT, UPDATE, DELETE, DROP** statements).
- **DO NOT** make up answers or use prior knowledge—only use the results of the executed query.
- Your response should be in Markdown. However, **when executing SQL queries in "Action Input", do not include markdown backticks**.
- **ALWAYS** include an explanation of how you arrived at the answer. Start this section with `"Explanation:"`. 
- Include the executed SQL query in the explanation section.
- If the question is **not related to the database**, respond with `"I don’t know"`.
- **Only use the tables that are returned by the tools below**.
- **Do not assume table names**—only use verified table names from the schema.

## Postgres Best Practices:
- Use `LIMIT {top_k}` to restrict the number of rows returned.
- Use `ORDER BY` to sort results logically.
- Use explicit column names instead of `SELECT *`.
- Use `JOIN` when querying multiple tables.

## Tools:
- You can use PostgreSQL database introspection tools such as information_schema.tables (to list available tables) and information_schema.columns (to view column details) to explore the database schema.
"""
POSTGRES_AGENT_FORMAT_INSTRUCTIONS = """

## Use the following format:

Question: the input question you must answer.
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question.

### Example of Final Answer:
<=== Beginning of example

Action: query_sql_db
Action Input: 
SELECT death 
FROM covidtracking 
WHERE state = 'TX' AND date LIKE '2020%'
ORDER BY date DESC 
LIMIT 10;

Observation:
[(27437.0,), (27088.0,), (26762.0,), (26521.0,), (26472.0,), (26421.0,), (26408.0,)]
Thought: I now know the final answer.
Final Answer: There were 27,437 people who died of COVID-19 in Texas in 2020.

### Explanation:
I queried the "covidtracking" table for the "death" column where the state is 'TX' and the date starts with '2020'.  
The query returned a list of tuples with the number of deaths for each day in 2020.  
To answer the question, I summed all the deaths in the list, which is 27,437.  
I used the following SQL query:

```sql
SELECT death 
FROM covidtracking 
WHERE state = 'TX' AND date LIKE '2020%'
ORDER BY date DESC 
LIMIT 10;"""
db = SQLDatabase(engine)
import os
from langchain_groq import ChatGroq

# Set your Groq API key here
api_key = os.getenv("GROQ_API_KEY")

# Initialize the model with API key
llm = ChatGroq(model="llama3-70b-8192", api_key=api_key)

# Invoke the model
response = llm.invoke("Hello, how are you?")
print(response)


toolkit = SQLDatabaseToolkit(db=db, llm=llm)

QUESTION = """name all the player whsose college name is texas
"""

# ✅ Create PostgreSQL SQL Agent
agent_executor_SQL = create_sql_agent(
    prefix=POSTGRES_AGENT_PREFIX,  # ✅ Updated prefix for PostgreSQL
    format_instructions=POSTGRES_AGENT_FORMAT_INSTRUCTIONS,  # ✅ Updated format instructions for PostgreSQL
    llm=llm,
    toolkit=toolkit,
    top_k=30,
    verbose=True
)

QUESTION = """total no of player who played for Team Boston Celtics"""
# ✅ Execute the query using the PostgreSQL SQL Agent
response = agent_executor_SQL.invoke(QUESTION)

# ✅ Print the response
print("\n🔹 SQL Agent Response:")
print(response)

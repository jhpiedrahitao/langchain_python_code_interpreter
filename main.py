from dotenv import load_dotenv
from langchain.agents.agent_toolkits import create_python_agent, create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import PythonREPLTool, Tool
from langchain.agents import AgentType, initialize_agent

load_dotenv()


def main():
    print("start...")
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, 
        prefix="""You are an agent designed to write and execute python code to answer questions.
                You have access to a python REPL, which you can use to execute python code.
                If you get an error, debug your code and try again.
                Only use the output of your code to answer the question. 
                Do not try to install any libary You have all libraries and dependencies allready installed.
                You might know the answer without running any code, but you should still run the code to get the answer.
                If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
                """
    )

    # python_agent_executor.run(
    #     "generate and save in current working directory 3 qr codes that point to facebook"
    # )

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="episode_info.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    # csv_agent.run("How many columns are there in the file episode_info.csv")
    # csv_agent.run("wich writer wrote the most episodes? how many episodes did he write?")
    csv_agent.run("can you find any correlations betwen columns?, save the correlation matrix in the working directory as a fancy heatmap png")

    grand_agent = initialize_agent(
        tools=[
            Tool(
                name="PythonAgent",
                func=python_agent_executor.run,
                description="""useful when you need to transform natural language and write from it python and execute the python code,
                              returning the results of the code execution,
                            DO NOT SEND PYTHON CODE TO THIS TOOL""",
            ),
            Tool(
                name="CSVAgent",
                func=csv_agent.run,
                description="""useful when you need to answer question over episode_info.csv file,
                             takes an input the entire question and returns the answer after running pandas calculations""",
            ),
        ],
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )

    #grand_agent.run(
    #    """which is the number of chapters per season in episode_info.csv, 
    #    then generate and save in current working directory a qr code with the number of chapters per season"""
    #)


if __name__ == "__main__":
    main()

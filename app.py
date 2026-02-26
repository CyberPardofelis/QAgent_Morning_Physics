from agent.controller import run_agent
from rich import print
import torch

if __name__ == "__main__":


    print("[bold green]Quantum Dual-LLM Research Agent Ready[/bold green]")
    print("Modes: openai | deepseek | compare")

    mode = input("Select mode: ").strip()

    #'''
    while True:
        query = input("\nYou: ")

        result = run_agent(query, mode=mode)

        if isinstance(result, dict):
            print("\n[bold cyan]OpenAI:[/bold cyan]\n", result["OpenAI"])
            print("\n[bold magenta]DeepSeek:[/bold magenta]\n", result["DeepSeek"])
        else:
            print("\n[bold cyan]Agent:[/bold cyan]", result)
    #'''
import asyncio
from retrieval import retrieve_parallel, TOP_K
from conversation import add_to_history, clear_history
from llm import build_prompt, call_chat

def interactive_chat():
    print("Hybrid travel assistant. Type 'exit' to quit, 'clear' to reset.\nEnter your question.")
    
    while True:
        query = input("\nYou: ").strip()
        if not query or query.lower() in ("exit", "quit"):
            break
        
        if query.lower() == "clear":
            clear_history()
            print("Conversation cleared.")
            continue

        add_to_history("user", query)

        matches, graph_facts = asyncio.run(retrieve_parallel(query, top_k=TOP_K))
        prompt = build_prompt(query, matches, graph_facts)
        answer = call_chat(prompt)

        add_to_history("assistant", answer)

        print("\n=== Assistant ===\n")
        print(answer)
        print("\n================\n")

if __name__ == "__main__":
    interactive_chat()
 
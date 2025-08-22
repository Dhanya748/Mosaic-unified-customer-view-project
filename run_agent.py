from segmented_olist_agent.agent.agent_graph import build_graph
from langchain_core.messages import HumanMessage

def main():
    graph = build_graph()  # this is already a callable function
    print("\nLangGraph agent connected. Ask me about your Olist database!")
    print("Type 'quit' or 'exit' to stop.\n")


    messages = []

    while True:
        try:
            q = input(">> ").strip()
            if q.lower() in {"quit", "exit"}:
                print("Goodbye!")
                break
            if not q:
                continue

            # Add user's message to the history
            messages.append(HumanMessage(content=q))

            # Directly call graph, no .invoke()
            resp = graph({"messages": messages})

            # The final AI response is the last message in the output
            final_response = resp["messages"][-1]

            # Add AI's response to history for context in the next turn
            messages.append(final_response)

            print("\n=== Agent Response ===")
            print(final_response.content)
            print("======================\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}\n")

if __name__ == "__main__":
    main()

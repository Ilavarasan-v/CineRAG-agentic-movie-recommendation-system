"""
main.py
───────
CLI entry point — run the agent from the terminal without the UI.

Usage:
    python main.py
    python main.py "best sci-fi movies about AI"
"""

import sys
from agent.graph import run_agent


def main():
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = [
            "space exploration movie with stunning visuals",
            "a romantic comedy set in New York",
            "90s thriller with a mind-bending twist",
        ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"Query : {query}")
        print("-" * 60)

        result = run_agent(query)

        print(f"Answer:\n{result['answer']}")
        print(f"\nSources  : {', '.join(result['sources']) or 'none'}")
        print(f"Loops    : {result['loops']}")
        print(f"Context score: {result['score']}/10")


if __name__ == "__main__":
    main()

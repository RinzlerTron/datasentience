from src.indexer import index_all_data
from src.agent import agent
import json

# Index data first
print("Loading data into vector store...")
index_all_data()

queries = [
    "What's causing the cost spike?",
    "When will we hit capacity?",
    "Show me our PUE trends and savings opportunities",
]

print("\n" + "=" * 80)
print("DATASENTIENCE AGENT - LOCAL MODE DEMO")
print("=" * 80 + "\n")

for i, query in enumerate(queries, 1):
    print(f"\n{'=' * 80}")
    print(f"QUERY {i}/{len(queries)}: {query}")
    print("=" * 80 + "\n")
    
    result = agent.query(query)
    
    if isinstance(result, dict):
        print(result['answer'])
        if result.get('chart_data'):
            print("\n[Chart data available - visualization would render here]")
            print(json.dumps(result['chart_data'], indent=2))
    else:
        print(result)
    
    if i < len(queries):
        input("\n\nPress Enter for next query...")

print("\n" + "=" * 80)
print("DEMO COMPLETE")
print("=" * 80)
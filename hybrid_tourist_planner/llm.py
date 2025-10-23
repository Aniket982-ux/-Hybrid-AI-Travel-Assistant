import config
from google import genai
from google.genai import types
from conversation import get_conversation_history

def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build prompt from retrieval results."""
    system = (
        "You are a helpful travel assistant that plans trips for users by using semantic search results from a vector database and graph facts from a knowledge graph. Use these data sources to provide answers that are concise, accurate, and relevant to the user's query."
        "Use semantic search results to identify relevant places, attractions, or entities related to the query."
        "Use graph facts to understand relationships between entities and ensure itineraries are realistic and feasible within the user's timeframe."
        "Provide answers in a friendly, professional tone, keeping responses under 300-450 words without unnecessary elaboration."
        "Where possible, include 2–3 concrete itinerary steps or tips per location with practical details. Suggest how much time to allocate or highlight top must-see attractions. If relevant, include travel or pacing advice to help users plan their days realistically."
        "If exact information is not available from the provided data, say you dont know. Do not invent or guess."
        "If the user's query references previous conversation (like 'it', 'that', 'make it different'), use the get_conversation_history tool to retrieve context."
        "Cite node IDs when referencing any specific places, attractions, or entities along with their names."
    )
    
    # Format Pinecone vector search results with IDs, names, and relevance scores
    vec_context = []
    for m in pinecone_matches:
        meta = m["metadata"]
        snippet = f"- id: {m['id']}, name: {meta.get('name','')}, type: {meta.get('type','')}, score: {m.get('score')}"
        if meta.get("city"):
            snippet += f", city: {meta.get('city')}"
        vec_context.append(snippet)
    # Format Neo4j graph relationships (source → relation → target)
    graph_context = [
        f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}"
        for f in graph_facts
    ]

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content":
         f"User query: {user_query}\n\n"
         "Top semantic matches:\n" + "\n".join(vec_context[:10]) + "\n\n"
         "Graph facts:\n" + "\n".join(graph_context[:20]) + "\n\n"
         "Answer the user's question."}
    ]
    return prompt

def call_chat(prompt_messages):
    """Call Gemini with tool support."""
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    
    system_msg = prompt_messages[0]['content']
    user_msg = prompt_messages[1]['content']
    
    # Define conversation history tool for LLM to call when needed
    memory_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_conversation_history",
                description="Retrieve recent conversation. Use when user references previous messages.",
                parameters={"type": "object", "properties": {}}
            )
        ]
    )
    
    full_prompt = f"{system_msg}\n\n{user_msg}"
    
    # First call: LLM decides whether to use conversation history tool
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt,
        config=types.GenerateContentConfig(
            tools=[memory_tool],
            system_instruction=system_msg
        )
    )
    # Check if LLM called the tool to retrieve conversation context
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                if part.function_call.name == "get_conversation_history":
                    # Tool called: fetch history and regenerate response with context
                    history = get_conversation_history()
                    full_prompt_with_history = f"{system_msg}\n\n{user_msg}\n\nConversation history:\n{history}"
                    
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=full_prompt_with_history,
                        config=types.GenerateContentConfig(system_instruction=system_msg)
                    )
                    break
    
    return response.text

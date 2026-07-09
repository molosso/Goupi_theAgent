"""
Coeur de l'agent : outils + boucle de decision.
pip install openai ddgs
"""
 
import os
import json
from openai import OpenAI
from ddgs import DDGS
 
client = OpenAI(
    api_key=os.environ.get("GroqApi", "YOUAPIKEY"),
    base_url="https://api.groq.com/openai/v1",
)
 
MODEL = "llama-3.3-70b-versatile"
 
SYSTEM_PROMPT = (
    "Tu es un agent qui resout des problemes en appelant des outils. "
    "IMPORTANT: n'appelle qu'UN SEUL outil a la fois, avec des valeurs concretes "
    "(jamais un autre appel de fonction comme argument). "
    "Si une etape depend du resultat d'une etape precedente, attends ce resultat "
    "avant de faire l'appel suivant. "
    "Pour repondre a des questions d'actualite ou que tu ne connais pas avec certitude, "
    "utilise l'outil search_web."
)
 
# --- Outils ---
 
def add(a: float, b: float) -> float:
    return a + b
 
def multiply(a: float, b: float) -> float:
    return a * b
 
def search_web(query: str) -> str:
    """Cherche sur le web via DuckDuckGo et renvoie un resume des resultats."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
    except Exception as e:
        return f"Erreur de recherche: {e}"
 
    if not results:
        return "Aucun resultat trouve."
 
    formatted = []
    for r in results:
        formatted.append(f"- {r['title']}: {r['body']} (source: {r['href']})")
    return "\n".join(formatted)
 
 
AVAILABLE_TOOLS = {"add": add, "multiply": multiply, "search_web": search_web}
 
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Additionne deux nombres",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiplie deux nombres",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Cherche des infos actuelles sur le web (actualites, faits recents, etc.)",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]
 
 
def run_agent(user_message: str, on_tool_call=None):
    """
    Lance l'agent sur un message utilisateur.
    on_tool_call: callback optionnel appele a chaque appel d'outil,
                  utile pour afficher la progression dans une interface.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
 
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
        )
        msg = response.choices[0].message
 
        if msg.tool_calls:
            messages.append(msg)
            for call in msg.tool_calls:
                fn_name = call.function.name
                args = json.loads(call.function.arguments)
                result = AVAILABLE_TOOLS[fn_name](**args)
 
                if on_tool_call:
                    on_tool_call(fn_name, args, result)
 
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
                })
            continue
 
        return msg.content
 
 
if __name__ == "__main__":
    q = "Quelles sont les dernieres nouvelles sur l'IA aujourd'hui ?"
    print("Question:", q)
    print("Reponse:", run_agent(q, on_tool_call=lambda n, a, r: print(f"  [outil] {n}({a})")))
 
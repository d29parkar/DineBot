import requests
from bs4 import BeautifulSoup
from googlesearch import search
from chatbot.state import State
from chatbot.config import llm, slm  # Use a smaller LLM for extraction

def google_search(state: State) -> State:
    """Fetches top Google search results if no internal results are found, updating state."""
    user_query = state["input"]
    state["google_results"] = [url for url in search(user_query, num_results=3)]
    return state


def extract_page_content(url):
    """Fetches and extracts meaningful text from a webpage."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()

        # Parse HTML and extract paragraphs
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        extracted_text = "\n".join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50)

        # Return first 1000 characters to ensure relevant context
        return extracted_text if extracted_text else "Content not available."
    
    except Exception as e:
        return f"Could not extract content: {str(e)}"

def google_search(state: State) -> State:
    """Fetches Google search results and extracts structured restaurant-related entities."""
    user_query = state["input"]
    search_results = list(search(user_query, num_results=3))

    extracted_data = []
    content_snippets = []

    for url in search_results:
        content = extract_page_content(url)
        extracted_data.append(f"🔗 **[{url}]({url})**")
        content_snippets.append(f"Source: {url}\nExtracted Content:\n{content}")

    if content_snippets:
        # **LLM Call for Entity Extraction**
        entity_extraction_prompt = f"""
        You are an advanced entity extraction assistant. Below is some web content related to restaurants, their menus, and their ingredients using blogs or websites.
        
        **User Query**: "{user_query}"

        **Extracted Content**:
        {content_snippets}
        
        **Task**:
        1. Identify all **restaurant names** mentioned.
        2. Extract the **dishes/food items** each restaurant is associated with.
        3. Preserve semantic meaning:
           - If a restaurant is mentioned **positively** for a certain dish, highlight that.
           - If it is mentioned **negatively** (e.g., "this place does NOT have gluten-free options"), retain that information.
        4. Format the response as structured insights, using bullet points.
        5. Preserve the **source website/blog name** for each restaurant.

        **Output Format Example**:
        Source: {url}
        - **Restaurant A**: Offers gluten-free pasta, highly recommended.
        - **Restaurant B**: Popular spot, but does not offer gluten-free options.
        - **Restaurant C**: Known for vegan pizza and salads.

        **Now extract and format the response. Return only the structured insights, no extra text.**
        """

        structured_response = slm.invoke(entity_extraction_prompt).content.strip()
        state["google_results"] = extracted_data + [structured_response] 
    else:
        state["google_results"] = ["No relevant external sources found."]
    
    return state

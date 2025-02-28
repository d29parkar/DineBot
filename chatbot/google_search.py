import requests
import wikipedia
from bs4 import BeautifulSoup
from googlesearch import search
from chatbot.state import State
from chatbot.config import slm


def fetch_page_content(url):
    """Fetches and extracts meaningful text from a webpage."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return f"Could not fetch {url} (HTTP {response.status_code})"

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        extracted_text = "\n".join(
            p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50
        )

        return " ".join(extracted_text.split()) if extracted_text else None  # Skip if no content

    except Exception as e:
        return None  # Skip this result


def google_search(state: State) -> State:
    """Performs Google search, fetches top 3 results, summarizes based on user query and intent."""
    user_query = state["input"]
    intent = state["intent"]

    # Perform Google Search
    search_results = list(search(user_query, num_results=3))
    page_contents = [fetch_page_content(url) for url in search_results if fetch_page_content(url)]
    structured_summary = ""

    for page_content in page_contents:
    # Construct a single, elaborate prompt
        summary_prompt = f"""
        You are an advanced AI designed to extract key insights from a webpage based on a specific user query type and user's intent.
        Read the provided webpage content and generate a structured summary that captures the most relevant information according to the query type. Follow the corresponding guidelines:

        **User Query**: "{user_query}"
        **Intent**: "{intent}"

        **Extracted Web Content**:
        {page_content}

        ### **Task:**
        if the intent of query is Ingredient-Based Discovery (e.g., 'Which restaurants serve gluten-free pizza?':
        - Identify and list all relevant restaurants mentioned.
        - Extract key details such as name, location, dish recommendations, menu highlights, pricing, and dietary accommodations.
        - Provide direct links or references for further exploration.

        if the intent of query is Trending Insights & Explanations (e.g., 'Latest trends in desserts in San Francisco'):
        - Summarize emerging trends, popular ingredients, and new restaurant offerings.
        - Identify key influencers, chefs, or brands driving the trend.
        - Include sources or data points supporting the trend, like recent mentions in news articles, social media, or menu updates.

        if the intent of query is Historical or Cultural Context (e.g., 'History of sushi and best sushi restaurants nearby')
        - Extract important historical dates, key figures, and the cultural evolution of the dish.
        - Summarize any regional or stylistic variations.
        - Identify highly-rated or historically significant restaurants that serve the dish, including any notable chef contributions.

        if the intent of query is Comparative Analysis (e.g., 'Compare vegan restaurant prices in SF vs. Mexican restaurants')
        - Extract and compare relevant statistics, such as average menu prices, customer ratings, or portion sizes.
        - Provide cost-of-living context or external economic factors that may influence pricing.
        - If data is missing, suggest alternative ways to interpret the comparison, such as chef interviews or food critic reviews.

        if the intent of query is Menu Innovation & Flavor Trend (e.g., 'How has the use of saffron in desserts changed?')
        - Identify frequency and variations of the ingredient on menus over time.
        - Capture mentions in culinary blogs, food industry reports, or chef interviews.
        - Provide context on why the trend is rising or declining, including cultural influences or seasonal availability.

        Ensure the summary is structured, fact-based, and concise while maintaining clarity and relevance. If multiple pages are available, extract only the most critical insights to avoid redundancy.
        """

        # Generate summary using LLM
        structured_summary += slm.invoke(summary_prompt[:6000]).content.strip()

    # Store results in state
    state["google_results"] = {
        "search_results": search_results,
        "summaries": structured_summary,  # Storing the final structured summary
    }
    return state

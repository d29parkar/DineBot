from chatbot.langgraph_workflow import app  # Ensure 'app' is already compiled
from chatbot.state import State
import json
from PIL import Image
import io

def visualize_workflow():
    """Saves the LangGraph workflow structure as 'workflow.png' using PIL."""
    print("\n🔹 Generating LangGraph Workflow...")

    # Generate workflow image in memory
    image_data = app.get_graph().draw_mermaid_png()  

    # Convert image bytes into a PIL image
    image = Image.open(io.BytesIO(image_data))

    # Save image
    image.save("workflow.png") 
    print("✅ Workflow saved as 'workflow.png'. Open the file to view it.")

def test_workflow(user_input):
    """Runs a test input through the LangGraph workflow and prints the updated state."""
    print(f"\n🚀 Testing Input: {user_input}")

    # Initialize correct state
    state = State(
        input=user_input,
        intent="",
        structured_results=[],
        faiss_results=[],
        google_results=[],
        response=""
    )

    print("\n🛠️ Initial State:")
    print(json.dumps(state, indent=2))  # Pretty-print initial state

    # Run LangGraph workflow and get updated state
    updated_state = app.invoke(state)  # ✅ Directly get final updated state

    print("\n✅ Updated State After Running LangGraph Workflow:")
    print(json.dumps(updated_state, indent=2))  # ✅ Print the updated state

if __name__ == "__main__":
    visualize_workflow()  # Save the LangGraph workflow as a PNG
    test_workflow("Which restaurants serve gluten-free pasta in San Francisco?")

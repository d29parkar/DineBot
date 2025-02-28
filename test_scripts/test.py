from chatbot.get_response import get_response

# Sample user query
test_query = "Which restaurants serve gluten-free pasta in San Francisco?"

# Get response from the system
response = get_response(test_query)

# Print output
print("Final Response:\n", response)

import ui

def main():
    """Main function to run the chatbot app."""
    user_query = ui.setup_ui()
    ui.handle_chat(user_query)

# Ensures script runs only when executed directly (not when imported)
if __name__ == "__main__":
    main()

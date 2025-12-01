def main():
    """Example usage of the Gemini Vertex AI processor"""
    
    # Initialize processor
    processor = GeminiVertexAIProcessor("config.yaml")
    
    # Example 1: Process an image
    print("=== Processing Image ===")
    image_results = processor.process_file(
        "sample_image.jpg",
        question="What objects can you identify in this image?",
        prompt_type="default"
    )
    print(f"Image Analysis: {image_results['response']}")
    
    # Example 2: Process a document
    print("\n=== Processing Document ===")
    doc_results = processor.process_file(
        "sample_document.pdf",
        question="Summarize the main points of this document",
        prompt_type="detailed"
    )
    print(f"Document Summary: {doc_results['response']}")
    
    # Example 3: Process a spreadsheet
    print("\n=== Processing Spreadsheet ===")
    spreadsheet_results = processor.process_file(
        "sample_data.xlsx",
        question="Analyze the trends in this data",
        prompt_type="trends"
    )
    print(f"Data Analysis: {spreadsheet_results['response']}")
    
    # Example 4: Process text file with custom prompt
    print("\n=== Processing Text File ===")
    text_results = processor.process_file(
        "sample_text.txt",
        question="Extract all email addresses and phone numbers from this text",
        prompt_type="default"
    )
    print(f"Text Extraction: {text_results['response']}")

if __name__ == "__main__":
    main()

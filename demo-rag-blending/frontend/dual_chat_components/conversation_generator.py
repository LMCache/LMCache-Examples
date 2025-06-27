import random
import os

CONTEXTS_PATH = "../contexts"

def get_available_docs():
    """Dynamically get the list of available documentation files from CONTEXTS_PATH."""
    try:
        if os.path.exists(CONTEXTS_PATH):
            files = [f for f in os.listdir(CONTEXTS_PATH) if os.path.isfile(os.path.join(CONTEXTS_PATH, f))]
            return files
        else:
            print(f"Warning: CONTEXTS_PATH directory not found at {CONTEXTS_PATH}")
            # Empty list if directory not found
            return []
    except Exception as e:
        print(f"Error listing CONTEXTS_PATH files: {e}")
        return []

# Cache for loaded document contents
DOC_CONTENTS = {}

def load_document(filename):
    """Load a document from the CONTEXTS_PATH directory."""
    if filename in DOC_CONTENTS:
        return DOC_CONTENTS[filename]

    try:
        # Try to load from CONTEXTS_PATH
        file_path = os.path.join(CONTEXTS_PATH, filename)

        # Check if file exists
        if not os.path.exists(file_path):
            # Fallback to old location as last resort
            file_path = os.path.join("../workload-generator/long-contexts", filename)

        print(f"Attempting to load from: {file_path}")

        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # Process based on file type
        if ext in ['.pdf', '.docx', '.doc', '.odt']:
            # For non-text documents, just return a placeholder
            # In a real implementation, you would use appropriate libraries
            # like PyPDF2, python-docx, etc. to extract text
            content = f"[Document content from {filename}]\n\nThis is a {ext[1:]} document."
        else:
            # Default: treat as text file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

        # Cache the content
        DOC_CONTENTS[filename] = content
        print(f"Successfully loaded {filename} ({len(content)} bytes)")
        return content
    except Exception as e:
        print(f"Failed to load document {filename}: {e}")
        # Return a short message instead of error to avoid breaking the demo
        return f"Sample documentation for {filename}"

def get_doc_chunk(doc_content, chunk_size=500):
    """Get a coherent chunk of documentation."""
    if len(doc_content) <= chunk_size:
        return doc_content

    # Split by paragraphs
    paragraphs = doc_content.split("\n\n")
    chunk = ""

    # Start from a random paragraph
    start_idx = random.randint(0, max(0, len(paragraphs) - 5))

    # Add paragraphs until we reach our target size
    for i in range(start_idx, len(paragraphs)):
        if len(chunk) + len(paragraphs[i]) + 2 <= chunk_size:
            chunk += paragraphs[i] + "\n\n"
        else:
            break

    return chunk.strip()

def get_doc_topic(doc_name):
    """Return a topic description based on the doc name."""
    # Extract basename and remove extension
    base_name = os.path.basename(doc_name)
    name_without_ext = os.path.splitext(base_name)[0]

    # Replace hyphens and underscores with spaces for better readability
    clean_name = name_without_ext.replace("-", " ").replace("_", " ")

    # Return generic description
    return f"{clean_name} concepts and usage"

def generate_question_docs(round_num, doc_names):
    """Generate a question about any documentation"""
    # Create a generic question template
    doc_name_pairs = []
    for i in range(min(len(doc_names), 3)):
        for j in range(i+1, min(len(doc_names), 4)):
            doc_name_pairs.append((doc_names[i], doc_names[j]))

    if not doc_name_pairs:
        # Fallback if we don't have enough docs
        return f"In 50 words, can you explain the key concepts in the available documentation?"

    # Choose a random pair of docs
    doc1, doc2 = random.choice(doc_name_pairs)

    templates = [
        f"In 50 words, can you explain how to use {doc1} for data processing?",
        f"In 50 words, what's the difference between {doc1} and {doc2}?",
        f"In 50 words, how can I combine {doc1} and {doc2} for better productivity?",
        f"In 50 words, what are the most useful features of {doc1}?",
        f"In 50 words, how do I use {doc2} effectively?",
        f"In 50 words, can you show me examples of using {doc1} with {doc2}?",
        f"In 50 words, what are the performance considerations when using {doc1}?",
        f"In 50 words, how can I use {doc1} with patterns to find information?",
        f"In 50 words, what's the best way to automate tasks using {doc2}?",
        f"In 50 words, can you explain the different options in {doc1}?",
        f"In 50 words, how do I optimize work with {doc2}?",
        f"In 50 words, what are some advanced techniques for processing with {doc1} and {doc2}?",
        f"In 50 words, how can I use {doc1} for operations?",
        f"In 50 words, what's the correct syntax for using {doc1} with {doc2}?",
        f"In 50 words, what are the common features of {doc1} and {doc2}?"
    ]

    return random.choice(templates)

def generate_system_response_docs(question, doc_names):
    """Generate a simulated system response based on the question for any docs"""
    # Find which docs are mentioned in the question
    mentioned_docs = []
    for doc in doc_names:
        if doc.lower() in question.lower():
            mentioned_docs.append(doc)

    if not mentioned_docs:
        # If no docs are specifically mentioned, use the first two
        mentioned_docs = doc_names[:min(2, len(doc_names))]

    # Get up to two docs to focus on
    focus_docs = mentioned_docs[:min(2, len(mentioned_docs))]

    # Create a generic response about these docs
    if len(focus_docs) >= 2:
        doc1, doc2 = focus_docs[0], focus_docs[1]
        return f"""Based on the documentation, let me explain how {doc1} and {doc2} work together.

{doc1} is primarily used for {get_doc_topic(doc1)}. Key features include:
- Feature 1: Basic functionality
- Feature 2: Advanced options
- Feature 3: Integration capabilities

{doc2} focuses on {get_doc_topic(doc2)} with these highlights:
- Primary use case: Main purpose
- Configuration: How to set it up
- Advanced usage: Expert techniques

When combining {doc1} and {doc2}, you can create powerful workflows that leverage the strengths of both. A typical integration might look like:

```
# Example code or command
basic_operation_with_{doc1.replace('-', '_')}()
advanced_processing_with_{doc2.replace('-', '_')}()
```

This approach gives you the benefits of {doc1}'s {get_doc_topic(doc1)} capabilities while adding {doc2}'s strengths in {get_doc_topic(doc2)}.
"""
    else:
        # Just one doc to focus on
        doc = focus_docs[0]
        return f"""Based on the {doc} documentation, here are the key points:

{doc} is designed for {get_doc_topic(doc)}. The main components include:

1. Core functionality:
   - Basic operations
   - Standard workflows
   - Common patterns

2. Advanced features:
   - Extended capabilities
   - Performance optimization
   - Integration points

Here's a simple example of how to use {doc}:

```
# Example usage of {doc}
basic_{doc.replace('-', '_')}_operation()
advanced_{doc.replace('-', '_')}_feature()
```

The documentation provides comprehensive information about all these aspects, with detailed explanations of each component and its usage scenarios.
"""

def generate_conversation(num_rounds=3):
    """Generate a full conversation with documentation"""
    # Dynamically get available doc files
    available_docs = get_available_docs()

    # Ensure we have at least some docs for compatibility
    if len(available_docs) == 0:
        # If no docs available, create dummy entries
        available_docs = [f"doc{i+1}.txt" for i in range(5)]

    # Select a reasonable number of documentation files (min of 5, max of all available)
    max_docs = min(max(5, len(available_docs)), len(available_docs))
    selected_docs = random.sample(available_docs, max_docs)

    # Keep full document names WITH extensions
    doc_names = selected_docs.copy()

    # Load document contents
    doc_contents = []
    for doc in selected_docs:
        doc_contents.append(load_document(doc))

    # Create a system message
    doc_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(doc_names)])
    system_message = {
        "role": "system",
        "content": f"""You are a helpful assistant that answers questions about technical topics.
You have access to the following documentation:
{doc_list}

Always keep your responses concise, aiming for around 50 words or less.
Focus on the most essential information and be direct.
Please provide accurate and helpful responses based on this documentation."""
    }

    # Create a conversation
    conversation = [system_message]

    # Generate rounds of question-answer pairs
    for i in range(num_rounds):
        question = generate_question_docs(i+1, doc_names)
        conversation.append({"role": "user", "content": question, "simulated": True})

        response = generate_system_response_docs(question, doc_names)
        conversation.append({"role": "assistant", "content": response, "simulated": True})

    return conversation, *doc_names

if __name__ == "__main__":
    # Test the conversation generator
    conversation, doc1, doc2, doc3, doc4, doc5 = generate_conversation(3)
    print(f"Generated conversation about {doc1}, {doc2}, {doc3}, {doc4}, and {doc5}")
    for msg in conversation:
        print(f"\n[{msg['role']}]")
        print(msg['content'])
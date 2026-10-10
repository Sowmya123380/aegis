
from document_processor import (
    extract_key_information,
    compare_documents,
)

doc1 = """Project A
Budget: 50000
Manager: Ravi
Email: ravi@example.com"""

doc2 = """Project A
Budget: 60000
Manager: Ravi
Email: ravi@example.com"""

print("KEY INFORMATION:")
print(extract_key_information(doc1))

print("\nDOCUMENT COMPARISON:")
print(compare_documents(doc1, doc2))
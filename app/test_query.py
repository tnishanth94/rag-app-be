from query_handler import query_pdf

# Test a sample query
question = "What is Symphony?"
response = query_pdf(question)

print("User Question:", question)
print("AI Response:", response["answer"])

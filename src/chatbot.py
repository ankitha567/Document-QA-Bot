import cohere
import uuid

class Chatbot:
    def __init__(self, vectorstore, cohere_api_key: str):
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())
        # Use Cohere V2 client
        self.co = cohere.ClientV2(api_key=cohere_api_key)

    def respond(self, user_message: str) -> tuple:
        """
        Retrieves context chunks using vectorstore and generates an answer using Cohere V2.
        """
        try:
            # 1. Ask Pinecone to find relevant text chunks for the query
            retrieved_docs = self.vectorstore.retrieve(user_message)
            
            # 2. Extract and join text data from metadata dictionary lists
            context = "\n".join([doc['text'] for doc in retrieved_docs])
        except Exception as err:
            return f"Error retrieving documents from database: {str(err)}", []

        system_instructions = (
            "You are a helpful AI assistant. Answer the user's question using ONLY the provided text context. "
            "If the answer cannot be found in the context, politely say 'I cannot find the answer in the document.'\n\n"
            f"--- DOCUMENT CONTEXT ---\n{context}"
        )

        try:
            # 3. Call modern Chat V2 endpoint structure
            response = self.co.chat(
                model="command-a-plus-05-2026",  # latest supported model
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_message}
                ]
            )

            # FIX: Collect all text blocks safely
            answer_parts = []
            for block in response.message.content:
                if block.get("type") == "text" and "text" in block:
                    answer_parts.append(block["text"])
            answer = "\n".join(answer_parts).strip()

            if not answer:
                answer = "I cannot find the answer in the document."

            return answer, retrieved_docs

        except Exception as e:
            return f"Error generating response from LLM: {str(e)}", retrieved_docs

from fastapi import APIRouter
from pydantic import BaseModel
from app.controllers.llm_controller import LLMController

# Create router for LLM chat interactions
router = APIRouter(prefix="/chat", tags=["LLM Chat"])

# Create a Pydantic schema for the input text
class TextInput(BaseModel):
    text: str

# Initialize the LLM controller
llm_controller = LLMController()

@router.post("/")
def chat_with_llm(input: TextInput):
    # Send the text to the LLM controller and get a response
    response, tokens = llm_controller.get_response(input.text)
    return {"input": input.text, "response": response, "tokens": tokens}

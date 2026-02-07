from pydantic import BaseModel

# 1. INPUT SCHEMA
class ReviewRequest(BaseModel):
    text: str

# 2. OUTPUT SCHEMA
class ReviewResponse(BaseModel):
    sentiment: str
    priority: str
    flag: str
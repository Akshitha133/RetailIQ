from pydantic import BaseModel, EmailStr


class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    business_name: str
    category: str
    location: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class DecisionRequest(BaseModel):
    product: str
    review: str = ""
    store_id: str | None = None


class BusinessRecommendationRequest(BaseModel):
    business_id: int
    product: str
    review: str


class InventoryRequest(BaseModel):
    business_id: int
    product: str
    stock_quantity: int

class InventoryUpdateRequest(BaseModel):
    business_id: int
    product: str
    stock_quantity: int
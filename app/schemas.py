from pydantic import BaseModel

class ItemBase(BaseModel):
    name: str
    description: str | None = None
    price: int

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int

from typing import List, Dict
from app.schemas import Item, ItemCreate

# In-memory storage for items
items_db: List[Dict] = []

def create_item(item: ItemCreate) -> Item:
    new_id = len(items_db) + 1
    new_item = Item(id=new_id, **item.dict())
    items_db.append(new_item.dict())
    return new_item

def get_item(item_id: int) -> Item | None:
    for item in items_db:
        if item["id"] == item_id:
            return Item(**item)
    return None

def get_items(skip: int = 0, limit: int = 10) -> List[Item]:
    return [Item(**item) for item in items_db[skip: skip + limit]]

def update_item(item_id: int, item_data: ItemCreate) -> Item | None:
    for index, item in enumerate(items_db):
        if item["id"] == item_id:
            items_db[index] = {**item, **item_data.dict(), "id": item_id}
            return Item(**items_db[index])
    return None

def delete_item(item_id: int) -> bool:
    global items_db
    for index, item in enumerate(items_db):
        if item["id"] == item_id:
            del items_db[index]
            return True
    return False

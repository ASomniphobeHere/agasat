from fastapi import APIRouter, HTTPException
from app import crud, schemas

router = APIRouter(prefix="/items", tags=["items"])

@router.post("/", response_model=schemas.Item)
def create_item(item: schemas.ItemCreate):
    return crud.create_item(item=item)

@router.get("/{item_id}", response_model=schemas.Item)
def read_item(item_id: int):
    db_item = crud.get_item(item_id=item_id)
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item

@router.get("/", response_model=list[schemas.Item])
def read_items(skip: int = 0, limit: int = 10):
    return crud.get_items(skip=skip, limit=limit)

@router.put("/{item_id}", response_model=schemas.Item)
def update_item(item_id: int, item: schemas.ItemCreate):
    updated_item = crud.update_item(item_id=item_id, item_data=item)
    if updated_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return updated_item

@router.delete("/{item_id}")
def delete_item(item_id: int):
    if not crud.delete_item(item_id=item_id):
        raise HTTPException(status_code=404, detail="Item not found")
    return {"message": f"Item {item_id} deleted"}

from typing import Optional
from pydantic import BaseModel

class InvoiceExtraction(BaseModel):
    fecha: Optional[str] = None
    importe_total: Optional[float] = None
    emisor: Optional[str] = None
    concepto: Optional[str] = None

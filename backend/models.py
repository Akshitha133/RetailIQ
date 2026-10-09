from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)

    business = relationship(
        "Business",
        back_populates="owner",
        uselist=False,
        cascade="all, delete-orphan"
    )


class Business(Base):
    __tablename__ = "businesses"

    id = Column(Integer, primary_key=True, index=True)
    business_name = Column(String(150), nullable=False)
    category = Column(String(100), nullable=False)
    location = Column(String(150), nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    owner = relationship(
        "User",
        back_populates="business"
    )


class SalesRecord(Base):
    __tablename__ = "sales_records"

    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(
        Integer,
        ForeignKey("businesses.id"),
        nullable=False
    )
    sale_date = Column(String(50), nullable=False)
    product = Column(String(150), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)

    business = relationship("Business")


class InventoryRecord(Base):
    __tablename__ = "inventory_records"

    id = Column(Integer, primary_key=True, index=True)

    business_id = Column(
        Integer,
        ForeignKey("businesses.id"),
        nullable=False
    )

    product = Column(
        String(150),
        nullable=False
    )

    stock_quantity = Column(
        Integer,
        nullable=False
    )

    business = relationship("Business")
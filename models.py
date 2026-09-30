from sqlalchemy import Column, Integer, String, Text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    role = Column(String)  # 'user' or 'assistant'
    content = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

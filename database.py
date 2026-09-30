from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://jc1_user:jc1_pass@localhost/jc1_chat"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# print(
#     SessionLocal
# )
# from models import Base
# from database import engine

# Base.metadata.create_all(bind=engine)

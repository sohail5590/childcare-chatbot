from database import engine, SessionLocal
from models import State, County
from sqlalchemy import select
from database import Base

def seed():
    db = SessionLocal()

    # IMPORTANT: create tables BEFORE seeding
    Base.metadata.create_all(bind=engine)

    try:
        # Check if already seeded
        count = db.execute(select(State)).all()
        if count:
            print("🌱 DB already seeded.")
            return

        print("🌱 Seeding states and counties...")

        states = {
            "California": ["Los Angeles", "Orange", "San Diego"],
            "Kentucky": ["Jefferson", "Fayette"],
            "Ohio": ["Franklin", "Cuyahoga"],
            "New York": ["Kings", "Queens", "Nassau"]
        }

        for state_name, county_list in states.items():
            st = State(name=state_name)
            db.add(st)
            db.commit()
            db.refresh(st)

            for cname in county_list:
                ct = County(name=cname, state_id=st.id)
                db.add(ct)

        db.commit()
        print("🌱 Seed complete.")

    except Exception as e:
        print(f"Seed error: {e}")

    finally:
        db.close()


if __name__ == "__main__":
    seed()

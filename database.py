import sqlite3

FARMERS_DB = "farmers.db"
PLANNER_DB = "farm.db"


def create_farmer_db():
    conn = sqlite3.connect(FARMERS_DB)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS farmers(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            password TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def create_planner_db():
    conn = sqlite3.connect(PLANNER_DB)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            crop TEXT NOT NULL,
            location TEXT NOT NULL,
            date TEXT NOT NULL,
            task TEXT NOT NULL,
            contact_email TEXT,
            contact_phone TEXT
        )
        """
    )

    existing_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(tasks)").fetchall()
    }

    for column_name, column_type in (
        ("user_id", "INTEGER"),
        ("contact_email", "TEXT"),
        ("contact_phone", "TEXT"),
    ):
        if column_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE tasks ADD COLUMN {column_name} {column_type}"
            )

    conn.commit()
    conn.close()


def create_db():
    create_farmer_db()
    create_planner_db()


create_db()

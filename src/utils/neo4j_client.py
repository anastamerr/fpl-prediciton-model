"""
Neo4j driver utilities and lightweight health check helper.

Uses python-dotenv to pull credentials from environment if not explicitly passed.
"""

import os
from typing import Optional

from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv


def get_driver(
    uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Driver:
    """
    Create a Neo4j driver. If parameters are missing, fall back to environment
    variables populated from a .env file when present.
    """
    load_dotenv()
    uri = uri or os.getenv("NEO4J_URI")
    username = username or os.getenv("NEO4J_USERNAME")
    password = password or os.getenv("NEO4J_PASSWORD")

    if not all([uri, username, password]):
        raise ValueError("NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be set.")

    return GraphDatabase.driver(uri, auth=(username, password))


def verify_connection(driver: Driver) -> bool:
    """
    Simple connectivity check. Returns True on success, False otherwise.
    """
    try:
        with driver.session() as session:
            result = session.run("RETURN 1 AS ok").single()
            return bool(result and result["ok"] == 1)
    except Exception:
        return False

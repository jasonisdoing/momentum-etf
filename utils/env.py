def load_env_if_present() -> bool:
    """Load variables from a .env file if python-dotenv is available.

    Returns True if a .env file was found and loaded.
    """
    try:
        from dotenv import load_dotenv, find_dotenv  # type: ignore
    except Exception:
        return False

    env_path = find_dotenv()
    if not env_path:
        return False
    # Use override=True to refresh values in long-lived processes (e.g., Streamlit)
    return bool(load_dotenv(env_path, override=True))

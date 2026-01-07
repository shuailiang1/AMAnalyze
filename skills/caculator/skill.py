def run(expression: str) -> str:
    """
    Calculate a mathematical expression safely.
    """
    print('---------------------------------')
    print(expression)
    print('---------------------------------')
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

SYSTEM_COMMANDS = {
    "оператор": {"answer": None, "handoff": True},
    "человек": {"answer": None, "handoff": True},
    "выход": {"answer": "Диалог завершён. Если понадоблюсь — обращайтесь!", "handoff": False},
}

def check_rule(message: str) -> tuple[str | None, bool]:
    msg = message.lower().strip()

    words=msg.split()
    for key, val in SYSTEM_COMMANDS.items():
        if key in words:
            return val["answer"], val["handoff"]
            
    return None, False
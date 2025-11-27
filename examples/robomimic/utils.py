def get_language_instruction(env_name):
    if env_name == "NutAssemblySquare":
        return "pick up the square nut and place it on the square peg"
    else:
        raise NotImplementedError
# ======================== IMPORTS ========================


# ======================== CLASSES ========================
class Style:
    def __init__(self) -> None:
        # Styling other than colors
        self.RESET      = "\033[0m"
        self.BOLD       = "\033[01m"
        self.UNDERLINE  = "\033[4m"

        # Foreground colors
        self.FG_RED     = "\033[31m"
        self.FG_GREEN   = "\033[32m"
        self.FG_YELLOW  = "\033[33m"
        self.FG_BLUE    = "\033[34m"
        self.FG_MAGENTA = "\033[35m"
        self.FG_CYAN    = "\033[36m"
        self.FG_WHITE   = "\033[37m"

        # Background colors
        self.BG_RED     = "\033[41m"
        self.BG_GREEN   = "\033[42m"
        self.BG_YELLOW  = "\033[43m"
        self.BG_BLUE    = "\033[44m"
        self.BG_MAGENTA = "\033[45m"
        self.BG_CYAN    = "\033[46m"
        self.BG_WHITE   = "\033[47m"

style = Style()


# =================== UTILITY FUNCTIONS ===================
def load_config():
    import yaml
    with open("../config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config

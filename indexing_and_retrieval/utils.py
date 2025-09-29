# ======================== IMPORTS ========================


# =================== UTILITY FUNCTIONS ===================
def load_config():
    import yaml
    with open("../config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config

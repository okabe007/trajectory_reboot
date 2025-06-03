import configparser

def load_config_dict(path: str) -> dict:
    """
    .iniファイルを読み込んで、キー文字列と値(float/int/str)の辞書を返す
    """
    config = configparser.ConfigParser()
    config.read(path)

    const = {}
    for key in config["DEFAULT"]:
        val = config["DEFAULT"][key]
        try:
            if "." in val or "e" in val.lower():
                const[key] = float(val)
            else:
                const[key] = int(val)
        except ValueError:
            const[key] = val  # strのまま
    return const

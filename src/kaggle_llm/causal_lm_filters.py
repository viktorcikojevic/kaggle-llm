def begins_with_year(item):
    try:
        _ = int(item[:4])
    except:
        return False
    return len(item) > 5 and item[4] == " "


def is_year_in_something(item):
    try:
        _ = int(item[:4])
    except:
        return False
    return item[4:8] == " in "


def is_glossary(item):
    return "glossary" in item


def is_fiction(item):
    return "fiction" in item


def is_list_of(item):
    return item.startswith("list of")


def is_timeline_of(item):
    return item.startswith("timeline of")


def is_weather_of(item):
    return item.startswith("weather of")


def is_data_page(item):
    return "(data page)" in item

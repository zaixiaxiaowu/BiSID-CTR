ITEM_PROFILE_PROMPT = (
    "According to the following item metadata, extract key attributes, "
    "summarize the category, function, and usage scenarios, and output a canonical description."
)

USER_PROFILE_PROMPT = (
    "According to the following historical interactions, infer the user's stable interests "
    "and recent intent, and output an interest summary."
)


def format_item_metadata(item):
    keys = [
        "title",
        "genres",
        "brand/store",
        "store",
        "main_category",
        "categories",
        "price",
        "average_rating",
        "features",
        "description",
        "details",
    ]
    parts = []
    seen = set()
    for key in keys:
        alias = "store" if key == "brand/store" else key
        if alias in seen:
            continue
        value = item.get(alias) if key != "brand/store" else item.get("store")
        if value is None or value == "" or value == []:
            continue
        label = "brand/store" if key in {"brand/store", "store"} else key
        parts.append(f"{label}: {value}")
        seen.add(alias)
    if not parts:
        return str(item)
    return "\n".join(parts)


def format_user_history(history_items, reviews, max_items=20):
    lines = []
    for item, review in list(zip(history_items, reviews))[-max_items:]:
        title = (item or {}).get("title") or (review or {}).get("title") or ""
        genres = (item or {}).get("genres") or (item or {}).get("categories") or ""
        text = " ".join(str((review or {}).get("text") or "").split())[:400]
        rating = (review or {}).get("rating")
        bits = [title]
        if genres:
            bits.append(f"[{genres}]")
        if rating not in (None, ""):
            bits.append(f"rating={rating}")
        line = "- " + " ".join(str(x) for x in bits if x)
        if text:
            line += f". review: {text}"
        lines.append(line)
    if not lines:
        return "(no history)"
    return "\n".join(lines)

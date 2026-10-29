ITEM_PROFILE_PROMPT = (
    "According to the following item metadata, extract key attributes, "
    "summarize the category, function, and usage scenarios, and output a canonical description."
)

USER_PROFILE_PROMPT = (
    "According to the following historical interactions, infer the user's stable interests "
    "and recent intent, and output an interest summary."
)


def format_item_metadata(item):
    parts = [
        f"title: {item.get('title')}",
        f"brand/store: {item.get('store')}",
        f"main_category: {item.get('main_category')}",
        f"categories: {item.get('categories')}",
        f"price: {item.get('price')}",
        f"average_rating: {item.get('average_rating')}",
        f"features: {item.get('features')}",
        f"description: {item.get('description')}",
        f"details: {item.get('details')}",
    ]
    return "\n".join(parts)


def format_user_history(history_items, reviews, max_items=20):
    lines = []
    for item, review in list(zip(history_items, reviews))[-max_items:]:
        title = item.get("title") if item else ""
        text = (review or {}).get("text") or ""
        text = " ".join(str(text).split())[:400]
        lines.append(f"- {title}. review: {text}")
    if not lines:
        return "(no history)"
    return "\n".join(lines)

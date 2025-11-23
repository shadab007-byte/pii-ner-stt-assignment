import json
import random

PERSON_NAMES = ["ramesh sharma", "priyanka verma", "rohan mehta", "amit kumar", "sneha patel", "vikram singh", "anjali gupta", "raj malhotra", "deepika rao", "karan desai"]
CITIES = ["mumbai", "delhi", "bangalore", "chennai", "hyderabad", "pune", "kolkata", "ahmedabad", "jaipur", "lucknow"]
LOCATIONS = ["taj mahal", "india gate", "gateway of india", "red fort", "marine drive", "airport", "railway station", "mg road"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]

def gen_card():
    return ' '.join(str(random.randint(1000, 9999)) for _ in range(4))

def gen_phone():
    patterns = [
        ' '.join(str(random.randint(0, 9)) for _ in range(10)),
        str(random.randint(9000000000, 9999999999)),
        f"nine eight seven six five four three two one zero"[:random.randint(30,60)]
    ]
    return random.choice(patterns)

def gen_email(name=None):
    if not name:
        name = random.choice(PERSON_NAMES)
    domains = ["gmail dot com", "yahoo dot com", "outlook dot com"]
    return f"{name.replace(' ', ' dot ')} at {random.choice(domains)}"

def gen_date():
    return f"{random.randint(1,28):02d} {random.choice(MONTHS)[:3]} {random.randint(2020,2025)}"

def add_entity(text, entities, entity_text, label):
    start = len(text)
    text += entity_text
    end = len(text)
    entities.append({"start": start, "end": end, "label": label})
    return text, entities

def gen_utterance(uid):
    text = ""
    entities = []
    
    templates = []
    if random.random() < 0.3:
        templates.append(("my credit card number is ", gen_card(), "CREDIT_CARD"))
    if random.random() < 0.4:
        templates.append(("my phone number is ", gen_phone(), "PHONE"))
    if random.random() < 0.5:
        name = random.choice(PERSON_NAMES)
        templates.append(("my name is ", name, "PERSON_NAME"))
    if random.random() < 0.4:
        templates.append(("my email is ", gen_email(), "EMAIL"))
    if random.random() < 0.3:
        templates.append(("i will travel on ", gen_date(), "DATE"))
    if random.random() < 0.3:
        templates.append(("i live in ", random.choice(CITIES), "CITY"))
    if random.random() < 0.2:
        templates.append(("near ", random.choice(LOCATIONS), "LOCATION"))
    
    if not templates:
        templates.append(("my phone is ", gen_phone(), "PHONE"))
    
    random.shuffle(templates)
    for i, (intro, entity_text, label) in enumerate(templates):
        if i > 0:
            text += random.choice([" and ", " also ", " "])
        text += intro
        text, entities = add_entity(text, entities, entity_text, label)
    
    return {"id": f"utt_{uid:04d}", "text": text.strip(), "entities": entities}

# Generate data
train_data = [gen_utterance(i+1) for i in range(600)]
dev_data = [gen_utterance(i+10001) for i in range(120)]
test_data = [{"id": f"utt_{i+20001:04d}", "text": gen_utterance(i+20001)["text"]} for i in range(30)]

with open("data/train.jsonl", "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open("data/dev.jsonl", "w", encoding="utf-8") as f:
    for item in dev_data:
        f.write(json.dumps(item) + "\n")

with open("data/test.jsonl", "w", encoding="utf-8") as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")

print(f"Generated {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test")
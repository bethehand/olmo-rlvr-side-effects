"""Write heldout_prompts.jsonl: 150 new prompts in three categories for the per-item quality check of the DPO arms
(#2002 plan, step 4). Written by hand on 2026-10-23, before any arm's outputs were seen; none of them is an AlpacaEval,
IFEval or GSM8K prompt (build_heldout.py --check ALPACA_EVAL_JSON reports the nearest AlpacaEval instruction per prompt).

  short   50 questions with a short, checkable answer. `gold` lists acceptable answer strings; the question is whether the
          model over-answers (answer length) while staying correct.
  detail  50 how/why questions whose good answer must cover several specific points. `rubric` lists those points; the
          judge marks each as covered or not, so a shorter answer that drops points is caught.
  open    50 open-ended requests (advice, explanation, short writing) for the length-controlled pairwise comparison.
"""
import argparse, json, re

SHORT = [
    ("What is the capital of New Zealand?", ["wellington"]),
    ("How many minutes are there in a day?", ["1440", "1,440"]),
    ("Convert 10 kilometers to miles. A single number is fine.", ["6.2", "6.21", "6.214"]),
    ("Which planet is closest to the Sun?", ["mercury"]),
    ("What is the chemical symbol for sodium?", ["Na"]),
    ("Is 91 a prime number?", ["no", "not prime", "not a prime", "composite"]),
    ("Who wrote Pride and Prejudice?", ["jane austen", "austen"]),
    ("What is 15% of 240?", ["36"]),
    ("In which year did the Berlin Wall fall?", ["1989"]),
    ("What is the boiling point of water at sea level, in Celsius?", ["100"]),
    ("How many sides does a hexagon have?", ["6", "six"]),
    ("What does CPU stand for?", ["central processing unit"]),
    ("What is the past tense of the verb 'swim'?", ["swam"]),
    ("Which is larger, 2/3 or 3/5?", ["2/3", "\\frac{2}{3}", "two-thirds", "two thirds"]),
    ("What is the square root of 169?", ["13"]),
    ("What is the longest river in South America?", ["amazon"]),
    ("How many bytes are in a kibibyte?", ["1024", "1,024"]),
    ("What is the freezing point of water in Fahrenheit?", ["32"]),
    ("Which gas do plants take in from the air for photosynthesis?", ["carbon dioxide", "co2", "co₂"]),
    ("What is the plural of 'cactus'?", ["cacti", "cactuses"]),
    ("How do you say 'thank you' in Spanish?", ["gracias"]),
    ("How many degrees are in a right angle?", ["90"]),
    ("What is the deepest known point in the ocean called?", ["challenger deep", "mariana trench", "marianas trench"]),
    ("Which element has atomic number 1?", ["hydrogen"]),
    ("What is 7 cubed?", ["343"]),
    ("Botanically speaking, is a tomato a fruit or a vegetable?", ["fruit"]),
    ("What is the currency of Japan?", ["yen"]),
    ("How many players does a soccer team have on the field at once?", ["11", "eleven"]),
    ("What does HTTP stand for?", ["hypertext transfer protocol"]),
    ("What is the smallest prime number?", ["2", "two"]),
    ("What is the hex color code for pure white?", ["#ffffff", "ffffff"]),
    ("What is the next number in the sequence 2, 4, 8, 16, ...?", ["32"]),
    ("Which continent is Egypt in?", ["africa"]),
    ("What is the opposite of 'transparent'?", ["opaque"]),
    ("How many hours is 150 minutes?", ["2.5", "two and a half", "2 hours and 30 minutes", "2h30"]),
    ("What is the median of 3, 9, 4, 7, 5?", ["5"]),
    ("What keyboard shortcut copies selected text on Windows?", ["ctrl+c", "ctrl + c", "control+c", "ctrl-c"]),
    ("In Python, what does len(\"hello\") return?", ["5"]),
    ("Can the Great Wall of China be seen from the Moon with the naked eye?", ["no", "cannot", "can't", "not visible"]),
    ("What is the SI unit of force?", ["newton"]),
    ("Which blood type is the universal donor for red blood cell transfusions?", ["o negative", "o-", "o−", "type o neg"]),
    ("What is the capital of Canada?", ["ottawa"]),
    ("Round 3.14159 to two decimal places.", ["3.14"]),
    ("How many strings does a standard guitar have?", ["6", "six"]),
    ("What is the Latin-derived abbreviation for 'for example'?", ["e.g."]),
    ("Which organ pumps blood through the human body?", ["heart"]),
    ("How many days are there in a leap year?", ["366"]),
    ("What is 2 to the power of 10?", ["1024", "1,024"]),
    ("What is the chemical formula of table salt?", ["nacl"]),
    ("Which is heavier, a kilogram of feathers or a kilogram of steel?", ["same", "neither", "equal", "weigh the same"]),
]

DETAIL = [
    ("How do I set up SSH key authentication from my Mac to a Linux server?",
     ["generate a key pair (ssh-keygen)", "copy the public key to the server (ssh-copy-id or ~/.ssh/authorized_keys)",
      "file permissions (~/.ssh 700, authorized_keys 600)", "test the login", "optionally disable password login in sshd_config"]),
    ("Explain how to make a sourdough starter from scratch.",
     ["mix flour and water", "feed it daily, discarding part", "keep it at a warm room temperature", "takes about 5 to 7 days",
      "readiness signs: bubbles, doubling in volume, sour smell"]),
    ("What should I check before buying a used car?",
     ["vehicle history report (VIN)", "independent mechanic inspection", "test drive", "title, liens and ownership papers",
      "service records and mileage", "body, rust, tires, fluids"]),
    ("How does a bill become a law in the United States?",
     ["introduced in the House or Senate", "committee review", "passed by both chambers", "differences reconciled between chambers",
      "president signs or vetoes", "veto override by two-thirds"]),
    ("Explain how HTTPS protects a connection.",
     ["TLS handshake", "certificate verifies the server's identity (certificate authority)", "key exchange produces a shared session key",
      "data is encrypted in transit", "integrity protection against tampering"]),
    ("What are the steps to perform CPR on an adult?",
     ["check responsiveness and breathing, call emergency services", "chest compressions at 100 to 120 per minute, about 5 to 6 cm deep",
      "30 compressions to 2 breaths, or hands-only", "use an AED if available", "continue until help arrives or the person recovers"]),
    ("How do I prepare for a job interview at a software company?",
     ["research the company and product", "practice coding or system design questions", "prepare behavioral examples (STAR)",
      "prepare questions to ask", "logistics: setup, attire, timing"]),
    ("Explain the difference between TCP and UDP and when to use each.",
     ["TCP is connection-oriented, reliable, ordered", "UDP is connectionless with no delivery guarantee", "TCP has more overhead or latency",
      "TCP use cases such as web, email, file transfer", "UDP use cases such as streaming, gaming, DNS, VoIP"]),
    ("How do vaccines train the immune system?",
     ["present an antigen (weakened, inactivated, protein or mRNA-made)", "immune response produces antibodies",
      "memory B and T cells are formed", "faster, stronger response on real exposure", "boosters or waning immunity"]),
    ("Give me a plan to start running if I have never exercised.",
     ["start slowly, consider a medical check", "alternate walking and running intervals", "about three sessions a week",
      "increase gradually (roughly 10% a week)", "proper shoes", "rest and recovery days"]),
    ("How do I write a good commit message?",
     ["short summary line (about 50 characters)", "imperative mood", "blank line then a body", "body explains what and why",
      "reference the issue or ticket"]),
    ("How does compound interest work? Give an example.",
     ["interest is earned on principal plus accumulated interest", "formula A = P(1 + r/n)^(nt) or equivalent",
      "a worked numeric example", "comparison with simple interest or the effect of time"]),
    ("What are the main causes of inflation?",
     ["demand-pull", "cost-push", "monetary expansion or too much money supply", "expectations or wage-price spiral", "supply shocks"]),
    ("How do I safely jump-start a car with a dead battery?",
     ["both cars off, cables untangled", "red clamp to dead battery positive, then good battery positive",
      "black clamp to good battery negative, then to bare metal ground on the dead car", "start the good car, then the dead car",
      "remove clamps in reverse order"]),
    ("Explain what a hash table is and how collisions are handled.",
     ["array indexed by a hash function", "average O(1) lookup", "collisions handled by chaining", "or by open addressing (probing)",
      "load factor and resizing"]),
    ("How should I store different kinds of fresh produce to keep them longest?",
     ["refrigerate leafy greens and berries", "potatoes, onions and garlic in a cool dark place, not the fridge",
      "tomatoes, bananas, avocados at room temperature until ripe", "separate ethylene producers from sensitive produce",
      "do not wash berries until use"]),
    ("What happens during a solar eclipse, and why isn't there one every month?",
     ["the Moon passes between the Sun and Earth and casts a shadow", "total, partial and annular types",
      "the Moon's orbit is tilted about 5 degrees", "alignment happens only near the nodes", "must be a new moon"]),
    ("How do I plan a budget for the first time?",
     ["track income", "list fixed and variable expenses", "set category limits (for example 50/30/20)",
      "emergency fund or savings goal", "review and adjust monthly"]),
    ("Explain the process of photosynthesis.",
     ["chlorophyll absorbs light", "water is split, releasing oxygen", "carbon dioxide is fixed into glucose (Calvin cycle)",
      "ATP and NADPH carry energy", "takes place in chloroplasts"]),
    ("What should be in a basic first-aid kit?",
     ["adhesive bandages and gauze", "antiseptic wipes or solution", "adhesive tape", "tweezers and scissors", "gloves",
      "pain relievers or personal medications", "emergency contact information"]),
    ("How does a refrigerator keep food cold?",
     ["refrigerant cycles between liquid and gas", "compressor", "condenser releases heat outside", "expansion valve",
      "evaporator absorbs heat from inside"]),
    ("How do I troubleshoot a laptop that will not turn on?",
     ["check power adapter, outlet and charging light", "hard reset: hold the power button, remove the battery if possible",
      "check indicator lights or beeps", "try an external display", "remove peripherals", "seek repair if hardware fault"]),
    ("Explain how recursion works with a simple example.",
     ["a function calls itself", "base case", "recursive case", "example such as factorial or Fibonacci",
      "call stack and stack overflow risk"]),
    ("What are the key differences between stocks and bonds?",
     ["stock is ownership, bond is a loan", "returns: dividends and capital gains versus interest", "risk and volatility",
      "priority of claims in bankruptcy", "what moves their prices"]),
    ("How do I grow tomatoes on a balcony?",
     ["large container with drainage (around 20 liters or more)", "6 to 8 hours of sun", "consistent watering",
      "staking or support", "fertilizer", "choose a compact or determinate variety"]),
    ("How does the greenhouse effect work?",
     ["sunlight warms the surface", "the surface emits infrared", "greenhouse gases absorb and re-emit infrared",
      "natural effect versus human-enhanced", "main gases: CO2, methane, water vapor"]),
    ("What steps should I take right after a minor car accident?",
     ["get to safety, hazard lights", "check for injuries, call police if required", "exchange contact and insurance details",
      "take photos and notes", "notify your insurer", "do not admit fault"]),
    ("How do I make cold brew coffee at home?",
     ["coarse grind", "coffee-to-water ratio around 1:4 to 1:8", "steep 12 to 24 hours", "filter or strain",
      "dilute to serve", "keeps about one to two weeks refrigerated"]),
    ("Explain how a bicycle's gears work.",
     ["front chainrings and rear cogs", "gear ratio from teeth counts", "low gear for climbing, high gear for speed",
      "derailleur moves the chain", "cadence or pedaling effort"]),
    ("How does the Electoral College elect the US president?",
     ["538 electors", "each state gets electors equal to its congressional delegation", "most states winner-take-all",
      "270 needed to win", "House of Representatives decides if no majority"]),
    ("How can I improve my sleep quality?",
     ["consistent sleep schedule", "limit caffeine, alcohol and screens before bed", "cool, dark, quiet room",
      "regular exercise", "wind-down routine", "avoid large late meals"]),
    ("Explain Git branching and merging to a beginner.",
     ["a branch is a separate line of development", "create and switch branches", "commits on the branch do not affect main",
      "merge brings changes back (fast-forward or merge commit)", "merge conflicts and how to resolve them"]),
    ("What causes the seasons on Earth?",
     ["Earth's axial tilt of about 23.5 degrees", "not the distance from the Sun", "the tilted hemisphere gets more direct light and longer days",
      "opposite seasons in the two hemispheres", "solstices and equinoxes"]),
    ("How do I write a résumé for my first job?",
     ["contact information", "education", "relevant experience, projects or volunteering", "skills section",
      "keep it to one page and tailor it", "action verbs and concrete results"]),
    ("How do noise-cancelling headphones work?",
     ["microphones pick up ambient sound", "the headphones generate an inverted (anti-phase) wave", "destructive interference cancels the noise",
      "works best on steady low-frequency noise", "passive isolation also helps"]),
    ("What is the general process for getting a passport?",
     ["fill in the application form", "proof of citizenship and identity", "passport photo meeting requirements", "pay the fee",
      "submit in person or by mail", "processing time and expedited options"]),
    ("How do airplanes stay in the air?",
     ["lift from the wings", "thrust from the engines", "drag", "weight or gravity", "angle of attack or wing shape",
      "control surfaces"]),
    ("How do I handle a conflict with a coworker?",
     ["cool down and reflect first", "talk privately and directly", "focus on behavior and impact, not the person",
      "listen to their side", "agree on concrete next steps", "escalate to a manager if unresolved"]),
    ("Explain the water cycle.",
     ["evaporation", "condensation into clouds", "precipitation", "collection, runoff and infiltration into groundwater",
      "driven by solar energy"]),
    ("How should I prepare my home for a hurricane?",
     ["emergency kit: water, food, medicine, flashlight", "secure or board windows and bring in loose items",
      "know the evacuation zone and route", "charge devices and fill fuel", "protect important documents", "follow official alerts"]),
    ("How do credit scores work?",
     ["payment history", "credit utilization", "length of credit history", "new credit and inquiries", "credit mix",
      "score range such as 300 to 850"]),
    ("Explain how a neural network learns.",
     ["layers of weighted connections", "forward pass produces a prediction", "loss function measures error",
      "backpropagation computes gradients", "gradient descent updates the weights", "repeated over many examples"]),
    ("How do I descale a kettle or coffee machine?",
     ["vinegar or citric acid solution", "fill or run a cycle and let it sit (20 to 60 minutes)", "rinse several times with clean water",
      "repeat if needed", "do it about monthly"]),
    ("What are the symptoms of dehydration, and how is it treated?",
     ["thirst, dark urine, dizziness, fatigue, dry mouth", "drink water or oral rehydration solution", "replace electrolytes",
      "severe signs such as confusion or fainting need medical care"]),
    ("How does GPS determine your location?",
     ["satellites broadcast their position and time", "the receiver measures signal travel time to get distances",
      "trilateration from at least four satellites", "clock error correction", "accuracy factors or augmentation"]),
    ("How do I choose a good password and manage many of them?",
     ["long passwords or passphrases (12+ characters)", "unique password for each site", "use a password manager",
      "enable two-factor authentication", "avoid personal information and reuse"]),
    ("Explain supply and demand with an example.",
     ["demand falls as price rises", "supply rises as price rises", "equilibrium price where they meet",
      "a concrete example of a shift", "shortage or surplus when price is off equilibrium"]),
    ("How do I change a flat tire?",
     ["safe flat location, parking brake on", "loosen the lug nuts before jacking", "jack up the car", "remove the flat and mount the spare",
      "tighten nuts in a star pattern", "lower the car and torque the nuts", "check the spare's pressure"]),
    ("How does the stock market work, for a beginner?",
     ["companies sell shares to raise money", "exchanges match buyers and sellers", "prices set by supply and demand",
      "you trade through a broker", "diversification or index funds", "risk and long-term horizon"]),
    ("How does an electric car battery work, and how can I prolong its life?",
     ["lithium-ion cells, ions move between electrodes", "battery management system", "avoid staying at 0% or 100% (keep roughly 20 to 80%)",
      "limit frequent fast charging", "avoid heat extremes", "degradation over cycles"]),
]

OPEN = [
    "Suggest a weekend itinerary for a first-time visitor to Kyoto in autumn.",
    "I'm 35 and want to switch careers into data analysis. How should I approach it?",
    "Write a short, friendly email declining a meeting invitation because of a scheduling conflict.",
    "What are some good strategies for learning a new language as an adult?",
    "Help me decide between buying and renting a home in a city where I plan to stay about four years.",
    "Explain to a 10-year-old why the sea is salty.",
    "Give me feedback on this thesis statement: \"Social media is bad for teenagers.\"",
    "What should I consider when adopting a dog if I work full time?",
    "Draft a short toast for my best friend's wedding. We met in college.",
    "How can a small engineering team improve its code review process?",
    "Recommend a reading order for someone new to science fiction.",
    "I keep procrastinating on my thesis. What can I do?",
    "Compare electric and gas stoves for a home cook.",
    "Write a product description for a reusable water bottle.",
    "What are the pros and cons of working remotely?",
    "Plan a vegetarian dinner party menu for six people.",
    "Explain the Monty Hall problem and why the answer feels counterintuitive.",
    "How do I give constructive feedback to a colleague who gets defensive?",
    "Write a haiku about the first snow of winter.",
    "What is a sensible approach to saving for retirement in your twenties?",
    "Describe the plot of Romeo and Juliet in a few sentences.",
    "Suggest ways to make a long-distance relationship work.",
    "I have eggs, spinach, feta and bread. What can I cook tonight?",
    "What makes a good team leader?",
    "Rewrite this sentence to be clearer: \"Due to the fact that the meeting was postponed, we were unable to finalize the decision which had been pending.\"",
    "How can I make my presentations more engaging?",
    "What should a beginner know before investing in index funds?",
    "Explain what machine learning is to a business executive.",
    "Suggest a name and a tagline for a neighborhood bakery.",
    "How do I prepare my garden for winter?",
    "What are the arguments for and against a four-day work week?",
    "Write a limerick about a cat who loves coffee.",
    "How should I structure a 20-minute conference talk?",
    "Give me tips for taking better photos with a phone.",
    "What are common mistakes people make when negotiating a salary?",
    "Explain how to read a nutrition label.",
    "How can parents limit children's screen time without constant fights?",
    "Recommend three board games for a family with teenagers and explain why.",
    "I'm nervous about public speaking. Any advice?",
    "Summarize the main idea of evolution by natural selection.",
    "How do I pick a laptop for university?",
    "Write a two-sentence horror story.",
    "What is the best way to learn to cook if I have never cooked?",
    "Explain the difference between weather and climate.",
    "Help me write a polite message to a neighbor about their loud music at night.",
    "What are good habits for maintaining mental health while working from home?",
    "Suggest a 30-minute bodyweight workout for beginners.",
    "Explain why airlines overbook flights.",
    "How do I start journaling and keep it up?",
    "Design a simple weekly meal-prep routine for one person.",
]


def rows():
    out = []
    for i, (q, gold) in enumerate(SHORT, 1):
        out.append({"id": f"S{i:02d}", "category": "short", "prompt": q, "gold": gold})
    for i, (q, rubric) in enumerate(DETAIL, 1):
        out.append({"id": f"D{i:02d}", "category": "detail", "prompt": q, "rubric": rubric})
    for i, q in enumerate(OPEN, 1):
        out.append({"id": f"O{i:02d}", "category": "open", "prompt": q})
    return out


def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower())) - {"the", "a", "an", "of", "to", "in", "is", "and", "i", "what", "how", "do", "for", "my", "me"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="heldout_prompts.jsonl")
    ap.add_argument("--check", default=None, help="alpaca_eval.json: report the nearest AlpacaEval instruction (Jaccard on words) per prompt")
    args = ap.parse_args()
    rs = rows()
    assert len(rs) == 150 and len({r["id"] for r in rs}) == 150 and len({r["prompt"] for r in rs}) == 150
    with open(args.out, "w") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rs)} prompts to {args.out}: " + ", ".join(f"{c} {sum(r['category'] == c for r in rs)}" for c in ("short", "detail", "open")))
    if args.check:
        ae = [x["instruction"] for x in json.load(open(args.check))]
        aw = [words(x) for x in ae]
        hits = []
        for r in rs:
            w = words(r["prompt"])
            j, k = max(((len(w & a) / max(1, len(w | a)), k) for k, a in enumerate(aw)))
            hits.append((j, r["id"], ae[k][:90].replace("\n", " ")))
        hits.sort(reverse=True)
        print("nearest AlpacaEval instruction, top 12 by word overlap (1.0 = same word set):")
        for j, i, t in hits[:12]:
            print(f"  {j:.2f} {i}  <- {t}")


if __name__ == "__main__":
    main()

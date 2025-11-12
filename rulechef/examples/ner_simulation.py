"""
NER Demonstration (Simulation Mode)

Shows RuleChef's workflow and value without requiring API key.
Simulates what rules would be learned and how they'd perform.
"""

from rulechef import RuleChef, Task
from rulechef.core import Rule, RuleFormat
from rulechef.coordinator import SimpleCoordinator
import re

# =============================================================================
# Define NER Task
# =============================================================================

print("=" * 80)
print("NER DEMONSTRATION: Extract Organizations, People, and Locations")
print("=" * 80)

task = Task(
    name="Named Entity Recognition",
    description="Extract organizations, people, and locations from news articles",
    input_schema={"text": "str"},
    output_schema={"spans": "List[Span]"},
)

coordinator = SimpleCoordinator(trigger_threshold=10, verbose=True)
chef = RuleChef(task, dataset_name="ner_simulation", coordinator=coordinator)

# =============================================================================
# Training Examples
# =============================================================================

training_examples = [
    {
        "text": "Apple Inc. announced its new iPhone at an event in Cupertino, California. CEO Tim Cook presented the features.",
        "entities": [
            ("Apple Inc.", "ORG"),
            ("Cupertino", "LOC"),
            ("California", "LOC"),
            ("Tim Cook", "PERSON"),
        ],
    },
    {
        "text": "Microsoft Corporation partnered with OpenAI. Satya Nadella expressed enthusiasm at the Seattle headquarters.",
        "entities": [
            ("Microsoft Corporation", "ORG"),
            ("OpenAI", "ORG"),
            ("Satya Nadella", "PERSON"),
            ("Seattle", "LOC"),
        ],
    },
    {
        "text": "Tesla Motors, based in Austin, Texas, released earnings. Elon Musk discussed expansion plans in Europe.",
        "entities": [
            ("Tesla Motors", "ORG"),
            ("Austin", "LOC"),
            ("Texas", "LOC"),
            ("Elon Musk", "PERSON"),
            ("Europe", "LOC"),
        ],
    },
    {
        "text": "Google LLC launched a data center in Dublin, Ireland. Sundar Pichai emphasized renewable energy.",
        "entities": [
            ("Google LLC", "ORG"),
            ("Dublin", "LOC"),
            ("Ireland", "LOC"),
            ("Sundar Pichai", "PERSON"),
        ],
    },
    {
        "text": "Amazon Web Services expanded in Singapore and Tokyo. Jeff Bezos praised the engineering team.",
        "entities": [
            ("Amazon Web Services", "ORG"),
            ("Singapore", "LOC"),
            ("Tokyo", "LOC"),
            ("Jeff Bezos", "PERSON"),
        ],
    },
]

print(f"\nAdding {len(training_examples)} training examples...")
for i, example in enumerate(training_examples, 1):
    text = example["text"]
    entities = example["entities"]

    # Find each entity in text
    spans = []
    for entity_text, entity_type in entities:
        start = text.find(entity_text)
        if start != -1:
            spans.append(
                {"text": entity_text, "start": start, "end": start + len(entity_text)}
            )

    chef.add_example(input_data={"text": text}, output_data={"spans": spans})
    print(f"  [{i}] Added: {', '.join([e[0] for e in entities])}")

print(f"\nBuffer stats: {chef.buffer.get_stats()}")

# =============================================================================
# Simulate Learned Rules (What RuleChef Would Learn)
# =============================================================================

print("\n" + "=" * 80)
print("SIMULATED LEARNED RULES (What RuleChef learns from LLM)")
print("=" * 80)

# Manually create rules that RuleChef would learn
simulated_rules = [
    Rule(
        id="org_1",
        name="Corporate Suffixes",
        description="Match organization names with Inc., LLC, Corporation, etc.",
        format=RuleFormat.REGEX,
        content=r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc\.|LLC|Corporation|Corp\.|Motors|Services))",
        priority=9,
        confidence=0.85,
    ),
    Rule(
        id="person_1",
        name="Capitalized Names",
        description="Match person names (First Last format, both capitalized)",
        format=RuleFormat.REGEX,
        content=r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)(?=\s+(?:announced|expressed|discussed|emphasized|praised|said))",
        priority=8,
        confidence=0.80,
    ),
    Rule(
        id="loc_1",
        name="City, State/Country",
        description="Match locations in 'City, State/Country' format",
        format=RuleFormat.REGEX,
        content=r"\b([A-Z][a-z]+),\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        priority=8,
        confidence=0.82,
    ),
    Rule(
        id="loc_2",
        name="Standalone Locations",
        description="Match standalone capitalized location names",
        format=RuleFormat.REGEX,
        content=r"\b(Seattle|Singapore|Tokyo|Europe|Dublin|Austin)\b",
        priority=7,
        confidence=0.75,
    ),
]

# Add rules to dataset
chef.dataset.rules = simulated_rules

for i, rule in enumerate(simulated_rules, 1):
    print(f"\n{i}. {rule.name}")
    print(f"   Pattern: {rule.content}")
    print(f"   Confidence: {rule.confidence:.2f}, Priority: {rule.priority}")

# =============================================================================
# Test on New Examples
# =============================================================================

test_examples = [
    "Salesforce acquired Slack in a deal announced in San Francisco. Marc Benioff praised the team.",
    "Adobe Systems launched Creative Cloud updates in Las Vegas, Nevada.",
    "Oracle Corporation moved to Austin, Texas. Larry Ellison explained the decision.",
]

print("\n" + "=" * 80)
print("TESTING ON NEW EXAMPLES")
print("=" * 80)

for i, text in enumerate(test_examples, 1):
    print(f"\n{i}. Text: {text}\n")

    # Apply rules manually (simulating rule engine)
    all_spans = []
    for rule in simulated_rules:
        pattern = re.compile(rule.content)
        for match in pattern.finditer(text):
            span_text = match.group(0 if match.lastindex is None else 1)
            start = text.find(span_text, match.start())
            all_spans.append(
                {
                    "text": span_text,
                    "start": start,
                    "end": start + len(span_text),
                    "rule": rule.name,
                }
            )

    # Deduplicate spans
    unique_spans = []
    seen = set()
    for span in all_spans:
        key = (span["text"], span["start"])
        if key not in seen:
            seen.add(key)
            unique_spans.append(span)

    print(f"   Extracted {len(unique_spans)} entities:")
    for span in unique_spans:
        print(
            f"     - '{span['text']}' [{span['start']}:{span['end']}] (rule: {span['rule']})"
        )

# =============================================================================
# Value Analysis
# =============================================================================

print("\n" + "=" * 80)
print("VALUE PROPOSITION ANALYSIS")
print("=" * 80)

print(f"""
Dataset: {len(training_examples)} training examples
Rules Learned: {len(simulated_rules)} patterns

Performance (estimated):
  - Precision: ~85% (catches most entities correctly)
  - Recall: ~70% (misses some edge cases)
  - F1 Score: ~77% (good balance)

Cost Comparison (per 1000 documents):
  - LLM approach: $5-50 (depending on model)
  - Rule approach: $0.00 (after initial learning)
  - Learning cost: $0.50-2 (one-time)

Speed Comparison:
  - LLM: 500-2000ms per document
  - Rules: <1ms per document (500-2000x faster!)

Break-even Point:
  - Rules pay for themselves after ~20-100 documents
  - At 1M documents: Save $5,000-50,000 and 6-23 days of processing time

When to Use RuleChef:
  ✓ High-volume extraction (1000s+ documents)
  ✓ Consistent patterns (company names, dates, locations)
  ✓ Real-time requirements (low latency needed)
  ✓ Cost-sensitive applications
  ✓ Privacy-sensitive data (rules run locally)

When to Use LLM Directly:
  ✗ One-off extractions (<100 documents)
  ✗ Highly variable input formats
  ✗ Complex reasoning required
  ✗ Rapidly changing requirements
""")

print("=" * 80)
print("\nTo run with real LLM learning:")
print("  export OPENAI_API_KEY='your-key'")
print("  python rulechef/examples/ner_demo.py")
print("=" * 80)

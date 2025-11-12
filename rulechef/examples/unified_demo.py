"""Demo of unified learning system - works with LLM observations and human input"""

from rulechef import RuleChef, Task
from rulechef.coordinator import SimpleCoordinator

# =============================================================================
# Setup
# =============================================================================

# Define task
task = Task(
    name="Q&A",
    description="Extract answer spans from questions about text",
    input_schema={"question": "str", "context": "str"},
    output_schema={"spans": "List[Span]"},
)

# Optional: Custom coordinator with different thresholds
coordinator = SimpleCoordinator(
    trigger_threshold=5,  # Learn after 5 examples (instead of 50)
    correction_threshold=2,  # Or 2 corrections
    verbose=True,
)

# Create RuleChef
chef = RuleChef(
    task,
    dataset_name="unified_demo",
    coordinator=coordinator,  # Optional: use custom coordinator
)

# =============================================================================
# Mode 1: Standalone (traditional - human examples only)
# =============================================================================

print("=" * 60)
print("MODE 1: Standalone - Traditional human-labeled examples")
print("=" * 60)

# Add human examples directly
chef.add_example(
    {"question": "When was it built?", "context": "The tower was built in 1889."},
    {"spans": [{"text": "1889", "start": 25, "end": 29}]},
)

chef.add_example(
    {"question": "What year?", "context": "Founded in 1995, it grew rapidly."},
    {"spans": [{"text": "1995", "start": 11, "end": 15}]},
)

print("✓ Added 2 human examples")
print(f"  Buffer stats: {chef.get_buffer_stats()}")

# Add more to reach threshold
for i in range(3):
    chef.buffer.add_human_example(
        {"question": f"Test {i}", "context": "Answer is here"},
        {"spans": [{"text": "here", "start": 10, "end": 14}]},
    )

print(f"✓ Buffer now has {chef.buffer.get_stats()['new_examples']} examples")

# Manual trigger
if chef.trigger_manual_learning():
    print("✓ Learning triggered successfully")
else:
    print("✗ Not ready to learn yet")

# =============================================================================
# Mode 2: Middleware (LLM observation)
# =============================================================================

print("\n" + "=" * 60)
print("MODE 2: Middleware - Observe LLM interactions")
print("=" * 60)

# NOTE: This requires OpenAI client
# Install with: pip install openai

try:
    from openai import OpenAI

    # Create OpenAI client
    # openai_client = OpenAI()  # Uncomment if you have API key

    # Start observing (wraps client)
    # wrapped_client = chef.start_observing(
    #     openai_client,
    #     auto_learn=True,  # Automatically trigger learning
    #     check_interval=10,  # Check every 10 seconds
    # )

    # Use wrapped client normally
    # response = wrapped_client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{
    #         "role": "user",
    #         "content": "Question: When was it built?\nContext: Built in 1990"
    #     }]
    # )

    # RuleChef observes the call and adds to buffer
    # When buffer reaches threshold, auto-triggers learning

    # Later: stop observing and use rules only
    # chef.stop_observing()

    print("✓ OpenAI available (code commented out - uncomment to use)")
    print("  Middleware mode allows observing LLM calls automatically")

except ImportError:
    print("✗ OpenAI not installed")
    print("  Install with: pip install openai")
    print("  Middleware mode requires OpenAI-compatible client")

# =============================================================================
# Mode 3: Hybrid (LLM + Human corrections)
# =============================================================================

print("\n" + "=" * 60)
print("MODE 3: Hybrid - LLM observations + Human corrections")
print("=" * 60)

# Simulate LLM observations
for i in range(3):
    chef.buffer.add_llm_observation(
        input_data={"question": f"Q{i}", "context": f"Context {i}"},
        output_data={"spans": [{"text": f"answer{i}", "start": 0, "end": 7}]},
    )

print("✓ Added 3 LLM observations")

# Add human corrections (high-value!)
chef.buffer.add_human_correction(
    input_data={"question": "When?", "context": "Built in 1990"},
    expected_output={"spans": [{"text": "1990", "start": 9, "end": 13}]},
    actual_output={"spans": [{"text": "Built", "start": 0, "end": 5}]},  # Wrong!
)

chef.buffer.add_human_correction(
    input_data={"question": "Where?", "context": "Located in Paris"},
    expected_output={"spans": [{"text": "Paris", "start": 11, "end": 16}]},
    actual_output={"spans": []},  # Missed it!
)

print("✓ Added 2 human corrections")
print(f"  Buffer stats: {chef.get_buffer_stats()}")

# Coordinator decides: 2 corrections >= threshold (2), so should learn
decision = chef.coordinator.should_trigger_learning(chef.buffer, chef.dataset.rules)
print("\nCoordinator decision:")
print(f"  Should learn: {decision.should_learn}")
print(f"  Reasoning: {decision.reasoning}")
print(f"  Strategy: {decision.strategy}")

# =============================================================================
# Coordinator Swapping (Simple -> Agentic)
# =============================================================================

print("\n" + "=" * 60)
print("FUTURE: Swappable Coordinators")
print("=" * 60)

print("Current: SimpleCoordinator (heuristic-based)")
print(f"  Trigger threshold: {chef.coordinator.trigger_threshold}")
print(f"  Correction threshold: {chef.coordinator.correction_threshold}")

print("\nFuture: AgenticCoordinator (LLM-powered)")
print("  - Analyzes buffer patterns intelligently")
print("  - Decides optimal timing for learning")
print("  - Adapts strategy based on data characteristics")
print("  - Drop-in replacement (same interface!)")

# To upgrade later:
# from rulechef.coordinator import AgenticCoordinator
# chef.coordinator = AgenticCoordinator(client)
# # Everything else stays the same!

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("✓ Unified learning system supports:")
print("  1. Standalone mode (human examples)")
print("  2. Middleware mode (LLM observation)")
print("  3. Hybrid mode (both sources)")
print("  4. Swappable coordinators (simple -> agentic)")
print("\n✓ Backwards compatible - existing code works unchanged!")

#!/usr/bin/env python3
"""Demo pipeline script that wires TokenizerNode -> EmbedderNode.

This script demonstrates a simple pipeline that:
1. Creates a TextModel with the text "Maida makes tiny AI shine."
2. Processes it through TokenizerNode -> EmbedderNode
3. Prints Ether.summary() after each hop

The script is designed to run in <500ms on a 4-core laptop and demonstrates
the basic Ether envelope flow between nodes.
"""

import time

from demo.nodes.embedder import EmbedderNode
from demo.nodes.tokenizer import TokenizerNode
from ether import Ether, TextModel


def print_summary(eth: Ether, step_name: str) -> None:
    """Print a formatted summary of an Ether envelope.

    Args:
        eth: The Ether envelope to summarize
        step_name: The name of the processing step
    """
    summary = eth.summary()
    print(f"\n=== {step_name} ===")
    print(f"Kind: {summary['kind']}")
    print(f"Schema Version: {summary['schema_version']}")
    print(f"Payload Keys: {summary['payload_keys']}")
    print(f"Metadata Keys: {summary['metadata_keys']}")
    print(f"Attachments: {summary['attachments']}")
    print(f"Source Model: {summary['source_model']}")

    # Print lineage information if present
    if "lineage" in eth.metadata:
        print(f"Lineage Length: {len(eth.metadata['lineage'])}")
        for i, entry in enumerate(eth.metadata["lineage"]):
            print(f"  {i+1}. {entry['node']} v{entry['version']} at {entry['ts']}")


def main() -> None:
    """Run the demo pipeline."""
    print("MaiEther Demo Pipeline")
    print("=" * 50)

    # Step 1: Create TextModel with the specified text
    print("\nStep 1: Creating TextModel")
    text_model = TextModel(text="Maida makes tiny AI shine.")
    print(f"Text: {text_model.text}")

    # Convert to Ether envelope
    eth = Ether(text_model)
    print_summary(eth, "Initial TextModel -> Ether")

    # Step 2: Process through TokenizerNode
    print("\nStep 2: Processing through TokenizerNode")
    tokenizer = TokenizerNode()
    tokenized_eth = tokenizer.process(eth)
    print_summary(tokenized_eth, "TokenizerNode Output")

    # Step 3: Process through EmbedderNode
    print("\nStep 3: Processing through EmbedderNode")
    embedder = EmbedderNode()
    final_eth = embedder.process(tokenized_eth)
    print_summary(final_eth, "EmbedderNode Output (Final)")

    # Step 4: Verify final envelope properties
    print("\nStep 4: Verification")
    print(f"Final envelope kind: {final_eth.kind}")
    print(f"Final envelope schema_version: {final_eth.schema_version}")

    # Verify acceptance criteria
    assert final_eth.kind == "embedding", f"Expected kind='embedding', got {final_eth.kind}"
    assert final_eth.schema_version == 1, f"Expected schema_version=1, got {final_eth.schema_version}"

    print("✅ All acceptance criteria met!")
    print("\nDemo completed successfully.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time) * 1000:.2f} ms")
